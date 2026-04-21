import torch
import numpy as np
import argparse
import ast
import os
import glob
import json
import yaml
from datetime import datetime
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from models_2_2_t2v.pipeline_TsAttn import WanTsAttnPipeline
from models_2_2_t2v.transformer_TsAttn import WanTsAttnTransformer3DModel

def ddp_setup():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    return rank, world_size

if __name__ == "__main__":
    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wanx")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--model_config", type=str, default="./configs/TS-Attn_t2v.yaml", help="The model for this experiment")
    parser.add_argument("--weights_path", type=str, default="[Your Path To Wan2.2-T2V-A14B-Diffusers]", help="The path to the weights")
    

    args = parser.parse_args()

    rank, world_size = ddp_setup()
    is_master = (rank == 0)

    dtype = torch.bfloat16

    weights_path = args.weights_path

    vae = AutoencoderKLWan.from_pretrained(weights_path, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTsAttnTransformer3DModel.from_pretrained(weights_path, torch_dtype=torch.bfloat16, subfolder='transformer')
    transformer_2 = WanTsAttnTransformer3DModel.from_pretrained(weights_path, torch_dtype=torch.bfloat16, subfolder='transformer_2')
    pipe = WanTsAttnPipeline.from_pretrained(weights_path, vae=vae, torch_dtype=dtype)
    pipe.transformer = transformer
    pipe.transformer_2 = transformer_2
    pipe.enable_model_cpu_offload(gpu_id=rank)

    model_config_path = args.model_config
    model_name = os.path.splitext(os.path.basename(model_config_path))[0]

    current_time = datetime.now().strftime("%Y%m%d-%H")
    
    model_type = "Wan2.2-T2V-A14B-Diffusers"

    output_folder = f"./output_videos/{model_type}/{model_name}_{current_time}_t2v"
    output_video_folder = os.path.join(output_folder, "videos")

    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    
    if is_master:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_video_folder, exist_ok=True)

    with open(model_config_path, 'r') as file:
        model_configs = yaml.safe_load(file)
    
    json_path = "./prompts/prompts_t2v.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        all_prompts = json.load(f)

    height = 480
    width = 832

    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    prompt_items = list(all_prompts.items())

    for idx, (name, data) in enumerate(prompt_items):
        if idx % world_size != rank:
            continue
            
        prompt = data["prompt"]
        print(f"Prompt: {prompt}")
        event_list = data["motion"]
        event_range = ast.literal_eval(data["event_range"])
        subject = data["subject"]

        if type(subject) is not list:
            subject = [subject]
    
        output = pipe(
            prompt=prompt,
            event_list=event_list,
            event_range=event_range,
            subject=subject,
            model_configs=model_configs,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=81,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            num_inference_steps=40,
            generator=torch.Generator().manual_seed(args.seed),
        )

        output = output.frames[0]
            
        output_path = os.path.join(output_video_folder, name)
        export_to_video(output, output_path, fps=16)

    if is_master:
        print("=== All processes finished ===")
