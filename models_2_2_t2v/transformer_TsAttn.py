
import math
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class TsAttnCrossAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    
    def get_event_video_mask(
        self,
        subject_visual_mask: Dict[str, torch.Tensor],
        proportions: list,
    ):
        subject_event_mask_list = {}
        subject_list = subject_visual_mask.keys()

        for subject in subject_list:
            mask = subject_visual_mask[subject] # Input: [1, N]
            B, N = mask.shape

            assert abs(sum(proportions) - 1.0) < 1e-6, "Proportions must sum to 1.0"

            counts = [round(p * N) for p in proportions]

            diff = sum(counts) - N
            counts[-1] -= diff

            splits = []
            start = 0
            for count in counts:
                end = start + count
                splits.append(mask[:, start:end])  # split [1, N] dim=1
                start = end
            subject_event_mask_list[subject] = splits
        return subject_event_mask_list

    
    @torch.no_grad()
    def get_dynamic_alpha(
        self,
        sim: torch.Tensor,
        alpha_min: float = 1.0,
        alpha_max: float = 1.5,
        eps: float = 1e-12
        ) -> torch.Tensor:

        p = F.softmax(sim, dim=-1)
        p_min = p.min(dim=-1, keepdim=True).values
        p_max = p.max(dim=-1, keepdim=True).values  # 0~1
        p =(p - p_min) / (p_max - p_min + eps)
        flat    = 1 - p  # 0~1

        alpha = alpha_min + flat * (alpha_max - alpha_min)  # [B, H, N_q, 1]

        return alpha

    
    def build_cross_attn_mask(
        self,
        subject_mask: Dict[str, torch.Tensor],
        subject_event_mask_list: Dict[str, list[torch.Tensor]],
        event_mask: Dict[str, torch.Tensor],
        cross_attn_mask: torch.Tensor,
        pre_cross_attn_map: torch.Tensor,
        mask_type: str,
    ):  
        subject_list = list(subject_event_mask_list.keys())
        event_list = list(event_mask.keys())

        # Soft Mask: subject region only attend to motion word
        if mask_type == "soft":
            for subject in subject_list:
                cross_attn_mask_list=[]
                subject_event_mask = subject_event_mask_list[subject]
                pre_cross_attn = pre_cross_attn_map[subject]

                start_idx = 0
                for i, subject_visual_mask in enumerate(subject_event_mask):
                    current_event = event_list[i]
                    sim = pre_cross_attn[:, :, start_idx:start_idx+subject_visual_mask.size(1), :]
                    max_attn = sim.max(dim=-1, keepdim=True).values  # [B, head, N, 1]
                    min_attn = sim.min(dim=-1, keepdim=True).values  # [B, head, N, 1]
                    mean_attn = sim.mean(dim=-1, keepdim=True)  # [B, head, N, 1]

                    final_mask = torch.zeros_like(sim)

                    alpha = self.get_dynamic_alpha(sim)

                    for j, other_event in enumerate(event_list):
                        mask = event_mask[other_event].unsqueeze(1).unsqueeze(2)
                        if other_event == current_event:
                            # subject token is positive
                            mask = mask | subject_mask[subject]
                            attn_bias = alpha*(-mean_attn + max_attn)* mask
                        else:
                            attn_bias = alpha*(-mean_attn + min_attn) * mask

                        final_mask += attn_bias

                    subject_visual_mask = subject_visual_mask.unsqueeze(1).unsqueeze(-1)
                    final_mask = final_mask * subject_visual_mask
                    cross_attn_mask_list.append(final_mask)
                    start_idx += subject_visual_mask.size(1)
                    
            cross_attn_mask = torch.cat(cross_attn_mask_list, dim=2)

        # Hard Mask: subject region only attend to corresponding motion word
        elif mask_type == "hard":
            for subject in subject_list:
                cross_attn_mask_list=[]
                subject_event_mask = subject_event_mask_list[subject]
                for i, mask in enumerate(subject_event_mask):
                    event = event_list[i]
                    cross_attn_event = mask.unsqueeze(-1) & event_mask[event].unsqueeze(-2)
                    cross_attn_mask_list.append(cross_attn_event)
                cross_attn_mask_list = torch.cat(cross_attn_mask_list, dim=1)
                cross_attn_mask |= cross_attn_mask_list
            
            background_mask = ~cross_attn_mask.any(dim=2)
            cross_attn_mask[background_mask] = True
        else:
            raise NotImplementedError(f"Invalid mask type: {mask_type}")
        
        return cross_attn_mask
    
    
    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        model_configs: Dict = None,
        timestep: int = None,
        use_control: bool = False,
        event_mask: Optional[Dict[str, torch.Tensor]] = None,
        event_range: Optional[list[float]] = None,
        subject_mask: Optional[Dict[str, torch.Tensor]] = None,
        subject_visual_mask: Optional[Dict[str, torch.Tensor]] = None,
        pre_cross_attn_map: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        USE_ENV_CONTROL = use_control
        
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        


        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if USE_ENV_CONTROL:
            subject_event_mask_list = self.get_event_video_mask(
                                subject_visual_mask=subject_visual_mask,
                                proportions = event_range if event_range is not None else [1.0 / len(event_mask)] * len(event_mask)
                                )

            batch_size = query.shape[0]
            n = query.shape[2]
            m = key.shape[2]
            cross_attn_mask = torch.zeros(batch_size, n, m, dtype=torch.bool, device=query.device)

            cross_attn_mask = self.build_cross_attn_mask(
                subject_mask=subject_mask,
                subject_event_mask_list=subject_event_mask_list,
                event_mask=event_mask,
                cross_attn_mask=cross_attn_mask,
                pre_cross_attn_map=pre_cross_attn_map,
                mask_type="soft",
            )

            attention_mask = cross_attn_mask


        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=TsAttnCrossAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    
    def get_attention_score(self, query, key, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None) -> torch.Tensor:
        """
        Scaled_dot_product_attention for obtaining attention score
        Equivalent to Efficient F.scaled_dot_product_attention
        """
        L, S = query.size(-2), key.size(-2)
        attn_bias = torch.zeros((L, S), dtype=query.dtype, device=query.device)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

        if attn_mask is not None:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

        unnorm_attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = unnorm_attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight, unnorm_attn_weight
    
    def erode_isolated(
        self,
        mask_1d: torch.Tensor,
        T: int = 21,
        H: int = 30,
        W: int = 52,
        k: int = 3,
    ) -> torch.Tensor:
        B, N = mask_1d.shape
        assert N == T * H * W, f"mask_1d shape: {mask_1d.shape}, T: {T}, H: {H}, W: {W}"

        grid = mask_1d.view(B, T, H, W).float()

        grid = grid.view(B * T, 1, H, W)

        kernel = torch.ones((1, 1, k, k), device=grid.device, dtype=grid.dtype)
        kernel[0, 0, k // 2, k // 2] = 0  # Set center to 0

        neighbor_sum = F.conv2d(grid, kernel, stride=1, padding=k // 2)

        eroded = (neighbor_sum > 0) & (grid > 0)

        eroded = eroded.view(B, T, H, W)

        mask_eroded = eroded.view(B, N).bool()

        return mask_eroded
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        model_configs: Dict,
        timestep: int,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        use_control: bool,
        subject_mask: Optional[Dict[str, torch.Tensor]] = None,
        event_mask: Optional[Dict[str, torch.Tensor]] = None,
        event_range: Optional[list[float]] = None,
        post_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        USE_ENV_CONTROL = (event_mask is not None) and (timestep[0] in model_configs["Env_args"]["time_step"]) and use_control

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)

        # Get Subject Layout
        subject_visual_mask = {}
        pre_cross_attn_map = {}

        if subject_mask:
            query_video = self.attn2.to_q(norm_hidden_states)
            key_text = self.attn2.to_k(encoder_hidden_states)

            subject_list = subject_mask.keys()

            query_video = query_video.unflatten(2, (self.attn2.heads, -1)).transpose(1, 2)
            key_text = key_text.unflatten(2, (self.attn2.heads, -1)).transpose(1, 2)

            for subject in subject_list:
                mask =  subject_mask[subject]

                attention_weights, unnorm_attention_weights = self.get_attention_score(query_video, key_text)
                attn_map = attention_weights.mean(dim=1)

                attn_min = attn_map.amin(dim=(1, 2), keepdim=True)
                attn_max = attn_map.amax(dim=(1, 2), keepdim=True)

                normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6) 

                subtoken_indices = mask.nonzero(as_tuple=True)[1] 
                selected_attn = normalized_attn[:, :, subtoken_indices]
                selected_attn = selected_attn.mean(dim=-1)
                
                
                sorted_attn, _ = torch.sort(selected_attn, descending=True)
                num_top50 = int(selected_attn.shape[1] * 0.5)
                top50_mean = sorted_attn[:, :num_top50].mean()

                bool_mask = selected_attn > top50_mean

                if model_configs["Env_args"]["use_erode"]:
                    bool_mask = self.erode_isolated(mask_1d=bool_mask, T=post_shape[0], H=post_shape[1], W=post_shape[2])

                subject_visual_mask[subject] = bool_mask
                pre_cross_attn_map[subject] = unnorm_attention_weights

        else:
            subject_visual_mask = None
            pre_cross_attn_map = None


        attn_output = self.attn2(hidden_states=norm_hidden_states,
                                 encoder_hidden_states=encoder_hidden_states,
                                 model_configs=model_configs,
                                 timestep=timestep,
                                 use_control=USE_ENV_CONTROL,
                                 event_mask=event_mask,
                                 event_range=event_range,
                                 subject_mask=subject_mask,
                                 subject_visual_mask=subject_visual_mask,
                                 pre_cross_attn_map=pre_cross_attn_map,
                                 )

        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTsAttnTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        model_configs: Dict,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        subject_mask: Optional[Dict[str, torch.Tensor]] = None,
        event_mask: Optional[Dict[str, torch.Tensor]] = None,
        event_range: Optional[list[float]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for i, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states=hidden_states, 
                    encoder_hidden_states=encoder_hidden_states,
                    model_configs=model_configs,
                    timestep=timestep,
                    temb=timestep_proj,
                    rotary_emb=rotary_emb,
                    use_control= (i % 2 != 0), #
                    subject_mask=subject_mask,
                    event_mask=event_mask,
                    event_range=event_range,
                    post_shape=(post_patch_num_frames, post_patch_height, post_patch_width)
                )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)