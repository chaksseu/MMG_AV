import argparse
import datetime
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from scipy.io.wavfile import write
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from tqdm import tqdm
import re
import traceback
import contextlib
import sys
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip

from accelerate import Accelerator

from scripts.evaluation.funcs import load_model_checkpoint
from utils.utils import instantiate_from_config
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like

from mmg_inference.auffusion_pipe_functions import (
    prepare_extra_step_kwargs, ConditionAdapter, import_model_class_from_model_name_or_path, Generator
)
from safetensors.torch import load_file

import csv

def load_accelerator_ckpt(model: torch.nn.Module, checkpoint_path: str):
    checkpoint = load_file(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model

def load_prompts(prompt_file: str) -> List[str]:
    """
    CSV 파일에서 'split'이 'test'인 행의 'caption'을 불러오는 함수
    """
    prompts = []
    try:
        with open(prompt_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('split') == 'test':
                    caption = row.get('caption', '').strip()
                    if caption:
                        prompts.append(caption)
    except Exception as e:
        print(f"Error reading prompt file: {e}")

    
    return prompts

import os

############################################################
# Utility Functions
############################################################
def save_videos(batch_tensors, savedir, base_filename, fps=10):
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(1, 0, 2, 3)  # t, c, h, w
        frames = video.numpy()
        frames = (frames + 1.0) / 2.0  # [0, 1] 범위로 정규화
        frames = (frames * 255).astype(np.uint8)
        frames_list = [frame for frame in frames]  # [t, c, h, w]
        
        # [h, w, c] 형태로 변환
        frames_list = [np.transpose(frame, (1, 2, 0)) for frame in frames_list]
        
        video_clip = ImageSequenceClip(frames_list, fps=fps)
        savepath = os.path.join(savedir, f"{base_filename}.mp4")
        video_clip.write_videofile(
            savepath,
            codec='libx264',
            audio=False,
            verbose=False,
            logger=None
        )


def sanitize_filename(text: str) -> str:
    text = text.replace(' ', '_')
    text = re.sub(r'[^A-Za-z0-9_\-]', '', text)
    return text


def split_prompts_evenly(prompts: List[str], num_splits: int) -> List[List[str]]:
    """
    전체 프롬프트를 num_splits개로 최대한 균등하게 나누는 함수
    """
    total = len(prompts)
    base = total // num_splits
    extra = total % num_splits
    subsets = []
    start = 0
    for i in range(num_splits):
        length = base + (1 if i < extra else 0)
        subsets.append(prompts[start:start+length])
        start += length
    return subsets

def initialize_ddim_sampler(model, schedule: str = "linear") -> dict:
    return {
        "model": model,
        "ddpm_num_timesteps": model.num_timesteps,
        "schedule": schedule,
        "counter": 0,
        "use_scale": model.use_scale
    }

def register_buffer(sampler_state: dict, name: str, attr: torch.Tensor) -> None:
    if isinstance(attr, torch.Tensor) and attr.device != torch.device("cuda"):
        attr = attr.to(torch.device("cuda"))
    sampler_state[name] = attr

def make_ddim_schedule(
    sampler_state: dict,
    ddim_num_steps: int,
    ddim_discretize: str = "uniform",
    ddim_eta: float = 0.0,
    verbose: bool = True
) -> None:
    model = sampler_state["model"]
    ddpm_num_timesteps = sampler_state["ddpm_num_timesteps"]
    use_scale = sampler_state["use_scale"]

    ddim_timesteps = make_ddim_timesteps(
        ddim_discr_method=ddim_discretize,
        num_ddim_timesteps=ddim_num_steps,
        num_ddpm_timesteps=ddpm_num_timesteps,
        verbose=verbose
    )

    alphas_cumprod = model.alphas_cumprod
    to_torch = lambda x: x.clone().detach().float().to(model.device)

    register_buffer(sampler_state, 'betas', to_torch(model.betas))
    register_buffer(sampler_state, 'alphas_cumprod', to_torch(alphas_cumprod))
    register_buffer(sampler_state, 'alphas_cumprod_prev', to_torch(model.alphas_cumprod_prev))

    if use_scale:
        register_buffer(sampler_state, 'scale_arr', to_torch(model.scale_arr))
        ddim_scale_arr = sampler_state['scale_arr'].cpu()[ddim_timesteps]
        register_buffer(sampler_state, 'ddim_scale_arr', ddim_scale_arr)
        ddim_scale_arr_prev = np.asarray([sampler_state['scale_arr'].cpu()[0]] +
                                         sampler_state['scale_arr'].cpu()[ddim_timesteps[:-1]].tolist())
        register_buffer(sampler_state, 'ddim_scale_arr_prev', ddim_scale_arr_prev)

    register_buffer(sampler_state, 'sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
    register_buffer(sampler_state, 'sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
    register_buffer(sampler_state, 'log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
    register_buffer(sampler_state, 'sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
    register_buffer(sampler_state, 'sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

    ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
        alphacums=alphas_cumprod.cpu(),
        ddim_timesteps=ddim_timesteps,
        eta=ddim_eta,
        verbose=verbose
    )
    register_buffer(sampler_state, 'ddim_sigmas', ddim_sigmas)
    register_buffer(sampler_state, 'ddim_alphas', ddim_alphas)
    register_buffer(sampler_state, 'ddim_alphas_prev', ddim_alphas_prev)
    register_buffer(sampler_state, 'ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

    sigmas_for_original_steps = ddim_eta * torch.sqrt(
        (1 - sampler_state['alphas_cumprod_prev']) / (1 - sampler_state['alphas_cumprod']) *
        (1 - sampler_state['alphas_cumprod'] / sampler_state['alphas_cumprod_prev'])
    )
    register_buffer(sampler_state, 'ddim_sigmas_for_original_num_steps', sigmas_for_original_steps)

    sampler_state['ddim_timesteps'] = ddim_timesteps
    sampler_state['ddim_num_steps'] = ddim_num_steps



@torch.no_grad()
def run_inference(
    accelerator,
    unet_model,
    video_model,
    prompt_file,
    savedir,
    bs,
    seed,
    unconditional_guidance_scale,
    num_inference_steps,
    height,
    width,
    frames,
    ddim_eta,
    fps
):
    try:        
        assert os.path.exists(prompt_file), f"Prompt file not found: {prompt_file}"
        all_prompts = load_prompts(prompt_file)


        ###all_prompts = all_prompts[:100]
        print("all_prompts length", len(all_prompts))

        # 전체 프롬프트를 num_processes에 맞게 균등 분배
        num_processes = accelerator.num_processes

        prompt_subsets = split_prompts_evenly(all_prompts, num_processes)


        # 현재 프로세스 인덱스에 해당하는 프롬프트 subset
        if accelerator.process_index < len(prompt_subsets):
            process_prompts = prompt_subsets[accelerator.process_index]
        else:
            process_prompts = []
        prompt_sublist = process_prompts
        

        # Set unique seed per process
        unique_seed = seed + accelerator.process_index
        seed_everything(unique_seed)
        device = accelerator.device
        generator = torch.Generator(device=device).manual_seed(unique_seed)


        if not prompt_sublist:
            accelerator.print(f"Process {accelerator.process_index}: No prompts to process.")
            #return

        do_video_cfg = unconditional_guidance_scale > 1.0

        assert (height % 16 == 0) and (width % 16 == 0), "Video dimensions must be multiples of 16!"
        latent_h, latent_w = height // 8, width // 8
        frames = video_model.temporal_length if frames < 0 else frames
        channels = video_model.channels

        # Create output dirs (main process only)
        if accelerator.is_main_process:
            os.makedirs(savedir, exist_ok=True)
        video_dir = savedir

        accelerator.wait_for_everyone()

        total_prompts = len(prompt_sublist)
        num_batches = (total_prompts + bs - 1) // bs

        # Initialize DDIM sampler for video
        sampler_state = initialize_ddim_sampler(video_model)
        make_ddim_schedule(sampler_state, ddim_num_steps=num_inference_steps, ddim_eta=ddim_eta, verbose=False)

        # Setup progress bar on main process
        if accelerator.is_main_process:
            pbar = tqdm(total=num_batches, desc="Generating", disable=not accelerator.is_main_process)
        else:
            pbar = None

        for batch_idx in range(num_batches):
            start_idx = batch_idx * bs
            end_idx = min(start_idx + bs, total_prompts)
            current_batch_size = end_idx - start_idx
            current_prompts = prompt_sublist[start_idx:end_idx]

            video_noise_shape = [current_batch_size, channels, frames, latent_h, latent_w]
            fps_tensor = torch.tensor([fps] * current_batch_size).to(device).long()

            # Get text embedding for video
            video_text_emb = video_model.get_learned_conditioning(current_prompts)
            
            video_cond = {"c_crossattn": [video_text_emb], "fps": fps_tensor}


            # Unconditional video
            cfg_scale = unconditional_guidance_scale
            uncond_type = video_model.uncond_type
            if cfg_scale != 1.0:
                if uncond_type == "empty_seq":
                    uncond_video_emb = video_model.get_learned_conditioning(current_batch_size * [""])
                elif uncond_type == "zero_embed":
                    c_emb = video_cond["c_crossattn"][0]
                    uncond_video_emb = torch.zeros_like(c_emb)
                else:
                    raise NotImplementedError("Unknown uncond_type")

                uncond_video_cond = {k: video_cond[k] for k in video_cond.keys()}
                uncond_video_cond.update({'c_crossattn': [uncond_video_emb]})
            else:
                uncond_video_cond = None

            # Initialize latents
            video_latents = torch.randn(video_noise_shape, device=device, generator=generator)

            timesteps = sampler_state['ddim_timesteps']
            total_steps = timesteps.shape[0]
            time_range = np.flip(timesteps)

            # Denoising loop
            for step_idx, video_step in enumerate(time_range):
                index = total_steps - step_idx - 1
                video_ts = torch.full((current_batch_size,), video_step, device=device, dtype=video_latents.dtype)

                # CFG for audio/video
                if do_video_cfg:
                    video_out_uncond = unet_model(
                        video_latents,
                        video_ts,
                        context=uncond_video_cond['c_crossattn'][0] if uncond_video_cond else None,
                        fps=uncond_video_cond['fps'] if uncond_video_cond else None
                    )

                video_out = unet_model(
                    video_latents,
                    video_ts,
                    context=video_cond['c_crossattn'][0],
                    fps=video_cond['fps']
                )

                if do_video_cfg:
                    video_out = video_out_uncond + cfg_scale * (video_out - video_out_uncond)

                # Video DDIM step
                use_original_steps = False
                alphas = video_model.alphas_cumprod if use_original_steps else sampler_state['ddim_alphas']
                alphas_prev = video_model.alphas_cumprod_prev if use_original_steps else sampler_state['ddim_alphas_prev']
                sqrt_oma = video_model.sqrt_one_minus_alphas_cumprod if use_original_steps else sampler_state['ddim_sqrt_one_minus_alphas']
                sigmas = sampler_state['ddim_sigmas_for_original_num_steps'] if use_original_steps else sampler_state['ddim_sigmas']

                is_video = (video_latents.dim() == 5)
                size = (current_batch_size, 1, 1, 1, 1) if is_video else (current_batch_size, 1, 1, 1)
                a_t = torch.full(size, alphas[index], device=device)
                a_prev = torch.full(size, alphas_prev[index], device=device)
                sigma_t = torch.full(size, sigmas[index], device=device)
                sqrt_oma_t = torch.full(size, sqrt_oma[index], device=device)

                pred_x0 = (video_latents - sqrt_oma_t * video_out) / a_t.sqrt()
                dir_xt = (1. - a_prev - sigma_t**2).sqrt() * video_out
                noise = sigma_t * torch.randn_like(video_latents) # , generator=generator

                if sampler_state['use_scale']:
                    scale_arr = video_model.scale_arr if use_original_steps else sampler_state['ddim_scale_arr']
                    scale_t = torch.full(size, scale_arr[index], device=device)
                    scale_arr_prev = video_model.scale_arr_prev if use_original_steps else sampler_state['ddim_scale_arr_prev']
                    scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
                    pred_x0 /= scale_t
                    video_latents = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
                else:
                    video_latents = a_prev.sqrt() * pred_x0 + dir_xt + noise


            # Decode video
            video_frames = video_model.decode_first_stage_2DAE(video_latents)

            # Save results (only main process should handle actual file operations)
            # 프로세스별로 구분하려면 파일명에 process_index 추가
            for i, prompt in enumerate(current_prompts):
                safe_prompt = sanitize_filename(prompt)
                sample_index = start_idx + i
                base_filename = f"{safe_prompt}_batch_{sample_index}_proc_{accelerator.process_index}_batch"
                # Save video
                video_filepath = os.path.join(video_dir, f"{base_filename}.mp4")
                single_video_frames = video_frames[i].unsqueeze(0)  
                save_videos(single_video_frames, video_dir, base_filename, fps=fps)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        accelerator.print(f"Process {accelerator.process_index}: Completed inference. Results saved in {savedir}.")

    except Exception as e:
        accelerator.print(f"Process {accelerator.process_index}: Encountered an error.")
        traceback.print_exc()



def main():
    parser = argparse.ArgumentParser(description="Video Generation Inference")
    # 기본 인자들
    parser.add_argument("--prompt_file", type=str, default="/workspace/processed_unseen_AVSync15/unseen_AVSync15_669.csv", help="CSV 파일 경로 (프롬프트 파일)")
    parser.add_argument("--savedir", type=str, default="/workspace/MMG_Inferencce_folder/0227_avsync_video_teacher", help="결과 비디오 저장 디렉토리")
    parser.add_argument("--bs", type=int, default=2, help="배치 사이즈")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="Unconditional guidance scale 값")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="DDIM inference 단계 수")
    parser.add_argument("--height", type=int, default=320, help="비디오 높이 (16의 배수)")
    parser.add_argument("--width", type=int, default=512, help="비디오 너비 (16의 배수)")
    parser.add_argument("--frames", type=int, default=40, help="생성할 프레임 수 (-1인 경우 모델의 기본값 사용)")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta 파라미터")
    parser.add_argument("--fps", type=float, default=12.5, help="비디오 FPS")

    # 모델 체크포인트 관련 인자
    parser.add_argument("--videocrafter_config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="UNet 모델 체크포인트 경로")
    parser.add_argument("--videocrafter_ckpt_path", type=str, default="scripts/evaluation/model.ckpt", help="Video 모델 체크포인트 경로")
    parser.add_argument("--video_lora_ckpt_path", type=str, default="/workspace/video_lora_training_checkpoints_0213/checkpoint-step-16384/model.safetensors"
, help="Video 모델 체크포인트 경로")


    args = parser.parse_args()

    # Accelerator 초기화
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float32

    video_config = OmegaConf.load(args.videocrafter_config)
    video_model = instantiate_from_config(video_config.model)
    state_dict = torch.load(args.videocrafter_ckpt_path)['state_dict']
    video_model.load_state_dict(state_dict, strict=False) # 로라 때문에 false
    video_model.to(device=device,dtype=dtype)
    video_unet = video_model.model.diffusion_model.eval()
    
    video_unet = load_accelerator_ckpt(video_unet, args.video_lora_ckpt_path)


    # 모델을 Accelerator 디바이스로 이동
    video_unet.to(accelerator.device)
    video_model.to(accelerator.device)

    # run_inference 실행
    run_inference(
        accelerator=accelerator,
        unet_model=video_unet,
        video_model=video_model,
        prompt_file=args.prompt_file,
        savedir=args.savedir,
        bs=args.bs,
        seed=args.seed,
        unconditional_guidance_scale=args.unconditional_guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        frames=args.frames,
        ddim_eta=args.ddim_eta,
        fps=args.fps
    )

if __name__ == "__main__":
    main()
