import json
import os
import numpy as np
import torch
from typing import Optional, List, Union, Tuple
from scipy.io.wavfile import write
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from tqdm import tqdm
import re
import traceback

import csv

from accelerate import Accelerator

from mmg_inference.auffusion_pipe_functions_copy_0123 import (
    encode_audio_prompt, prepare_extra_step_kwargs, ConditionAdapter, import_model_class_from_model_name_or_path, Generator
)

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
                # if row.get('split') == 'test':
                caption = row.get('caption', '').strip()
                if caption:
                    prompts.append(caption)
    except Exception as e:
        print(f"Error reading prompt file: {e}")

    
    return prompts


def sanitize_filename(text: str) -> str:
    text = text.replace(' ', '_')
    text = re.sub(r'[^A-Za-z0-9_\-]', '', text)
    return text


def denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200,
    min_value: float = 1e-5,
    power: float = 1
) -> torch.Tensor:
    assert len(data.shape) == 3, f"Expected 3D tensor, got shape {data.shape}"
    max_value = np.log(max_value)
    min_value = np.log(min_value)
    data = torch.flip(data, [1])
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    assert data.shape[0] == 3, "Spectrogram must have 3 channels"
    data = data[0]
    data = torch.pow(data, 1 / power)
    spectrogram = data * (max_value - min_value) + min_value
    return spectrogram

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

'''
def encode_audio_prompt(
    text_encoders: List[torch.nn.Module],
    tokenizers: List[AutoTokenizer],
    adapters: List[ConditionAdapter],
    max_length: int,
    dtype: torch.dtype,
    prompt: Union[List[str], str],
    device: torch.device,
    do_classifier_free_guidance: bool = False,
    prompt_embeds: Optional[torch.FloatTensor] = None
) -> torch.FloatTensor:

    assert len(text_encoders) == len(tokenizers), "Mismatched text_encoder and tokenizer counts."
    if adapters is not None:
        assert len(text_encoders) == len(adapters), "Mismatched text_encoder and adapter counts."

    def get_prompt_embeds(prompt_list, device):
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        all_embeds = []
        for p in prompt_list:
            cond_embs_list = []
            for idx in range(len(text_encoders)):
                input_ids = tokenizers[idx](p, return_tensors="pt").input_ids.to(device)
                encoded = text_encoders[idx](input_ids).last_hidden_state.to(device)

                # Pad/truncate
                if encoded.shape[1] < max_length:
                    pad_len = max_length - encoded.shape[1]
                    encoded = F.pad(encoded, (0,0,0,pad_len), value=0)
                else:
                    encoded = encoded[:, :max_length, :]

                # Condition adapter
                if adapters is not None:
                    encoded = adapters[idx](encoded)
                cond_embs_list.append(encoded)

            merged_embs = torch.cat(cond_embs_list, dim=1)
            all_embeds.append(merged_embs)
        return torch.cat(all_embeds, dim=0)

    if prompt_embeds is None:
        prompt_embeds = get_prompt_embeds(prompt, device)

    return prompt_embeds.to(dtype=dtype, device=device)
'''

############################################################
# Model and Pipeline Initialization
############################################################
def load_vocoder(model_path: str, device: torch.device, dtype: torch.dtype) -> Generator:
    return Generator.from_pretrained(model_path, subfolder="vocoder").to(device, dtype)

def load_vae(model_dir: str, device: torch.device, dtype: torch.dtype) -> Tuple[AutoencoderKL, VaeImageProcessor]:
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae").to(device, dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    return vae, image_processor

def load_text_encoders(
    model_dir: str,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[List[torch.nn.Module], List[AutoTokenizer], List[ConditionAdapter]]:
    condition_json_path = os.path.join(model_dir, "condition_config.json")
    with open(condition_json_path, "r") as f:
        condition_data = json.load(f)

    text_encoders, tokenizers, adapters = [], [], []
    with torch.no_grad():
        for item in condition_data:
            text_encoder_path = os.path.join(model_dir, item["text_encoder_name"])
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            tokenizers.append(tokenizer)

            text_encoder_cls = import_model_class_from_model_name_or_path(text_encoder_path)
            text_encoder = text_encoder_cls.from_pretrained(text_encoder_path).to(device, dtype)
            text_encoders.append(text_encoder)

            adapter_path = os.path.join(model_dir, item["condition_adapter_name"])
            adapter = ConditionAdapter.from_pretrained(adapter_path).to(device, dtype)
            adapters.append(adapter)

    return text_encoders, tokenizers, adapters

def load_audio_unet(model_dir: str, device: torch.device, dtype: torch.dtype) -> UNet2DConditionModel:
    return UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet").to(device, dtype)




@torch.no_grad()
def run_inference(
    accelerator,
    unet_model,
    vae,
    image_processor,
    text_encoder_list,
    adapter_list,
    tokenizer_list,
    seed = 1234,
    prompt_file = None,
    savedir = '0121_audio_teacher',
    bs = 4,
    pretrained_model_name_or_path = "auffusion/auffusion-full",
    duration = 3.2,
    guidance_scale = 7.5,
    num_inference_steps = 50,
    eta_audio = 0.0
):
    try:
        dtype = torch.float32
        
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

        audio_length = int(duration * 16000)
        #latent_time = int(fps * duration)
        latent_time = int(12.5 * duration)

        do_audio_cfg = guidance_scale > 1.0

        # Model directory
        model_dir = pretrained_model_name_or_path
        if not os.path.isdir(model_dir):
            model_dir = snapshot_download(model_dir)

        # Load audio components
        vocoder = load_vocoder(model_dir, device, dtype)        
        text_encoders, tokenizers, adapters = text_encoder_list, tokenizer_list, adapter_list
        
        audio_unet = unet_model
        

        # Create output dirs (main process only)
        if accelerator.is_main_process:
            os.makedirs(savedir, exist_ok=True)
        accelerator.wait_for_everyone()
        

        total_prompts = len(prompt_sublist)
        num_batches = (total_prompts + bs - 1) // bs

        # Setup progress bar on main process
        if accelerator.is_main_process:
            pbar = tqdm(total=num_batches, desc="Generating", disable=not accelerator.is_main_process)
        else:
            pbar = None


        # Audio scheduler
        audio_scheduler = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        audio_scheduler.set_timesteps(num_inference_steps, device=device)


        for batch_idx in range(num_batches):
            start_idx = batch_idx * bs
            end_idx = min(start_idx + bs, total_prompts)
            current_batch_size = end_idx - start_idx
            current_prompts = prompt_sublist[start_idx:end_idx]

            
            audio_timesteps = audio_scheduler.timesteps



            audio_prompt_embeds = encode_audio_prompt(
                text_encoder_list=text_encoders,
                tokenizer_list=tokenizers,
                adapter_list=adapters,
                tokenizer_model_max_length=77,
                dtype=dtype,
                prompt=current_prompts,
                device=accelerator.device
            )

            # Initialize audio latents
            audio_latent_shape = (current_batch_size, 4, 32, latent_time)
            audio_latents = randn_tensor(audio_latent_shape, generator=generator, device=device, dtype=audio_prompt_embeds.dtype)
            audio_latents *= audio_scheduler.init_noise_sigma

            extra_step_kwargs = prepare_extra_step_kwargs(audio_scheduler, generator, eta_audio)
            
            # (CFG 준비) 
            if do_audio_cfg:
                neg_audio_prompt_embeds = torch.zeros_like(
                    audio_prompt_embeds, dtype=audio_prompt_embeds.dtype, device=device
                )
            else:
                neg_audio_prompt_embeds = None

            # Denoising loop
            for audio_step in audio_timesteps:
                # CFG for audio
                if do_audio_cfg and neg_audio_prompt_embeds is not None:
                    audio_uncond_input = audio_scheduler.scale_model_input(audio_latents, audio_step)
                    audio_out_uncond = audio_unet(
                        audio_uncond_input,
                        audio_step,
                        encoder_hidden_states=neg_audio_prompt_embeds
                    )[0]

                audio_input = audio_scheduler.scale_model_input(audio_latents, audio_step)


                audio_out = audio_unet(
                    audio_input,
                    audio_step,
                    encoder_hidden_states=audio_prompt_embeds
                )[0]


                if do_audio_cfg:
                    audio_out = audio_out_uncond + guidance_scale * (audio_out - audio_out_uncond)

                # Update audio latents
                audio_latents = audio_scheduler.step(audio_out, audio_step, audio_latents, **extra_step_kwargs, return_dict=False)[0]


            # Decode audio

            audio_recon_image = vae.decode(audio_latents / vae.config.scaling_factor, return_dict=False)[0]
            do_denormalize = [True] * audio_recon_image.shape[0]
            

            audio_recon_image = image_processor.postprocess(audio_recon_image, output_type="pt", do_denormalize=do_denormalize)


            audios = []
            for img in audio_recon_image:
                spectrogram = denormalize_spectrogram(img)
                audio_waveform = vocoder.inference(spectrogram, lengths=audio_length)[0]
                audios.append(audio_waveform)


            # Save results (only main process should handle actual file operations)
            # 프로세스별로 구분하려면 파일명에 process_index 추가

            for i, prompt in enumerate(current_prompts):
                safe_prompt = sanitize_filename(prompt)
                sample_index = start_idx + i
                base_filename = f"{safe_prompt}_batch_{sample_index}_proc_{accelerator.process_index}_batch"

                # Save audio
                audio_filepath = os.path.join(savedir, f"{base_filename}.wav")
                write(audio_filepath, 16000, audios[i])
            
            
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        accelerator.print(f"Process {accelerator.process_index}: Completed inference. Results saved in {savedir}.")
        
        accelerator.wait_for_everyone()

    except Exception as e:
        accelerator.print(f"Process {accelerator.process_index}: Encountered an error.")
        traceback.print_exc()





import argparse
import os
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
import os
import sys

import argparse
import json
import random
from datetime import timedelta
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from diffusers import (
    PNDMScheduler,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL
)
from diffusers.image_processor import VaeImageProcessor


from einops import rearrange
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm
import wandb
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Audio Generation Inference")
    parser.add_argument("--prompt_file", type=str, default="/home/work/kby_hgh/audio_video_100_prompts.csv", help="CSV 파일 경로 (프롬프트 파일)")
    parser.add_argument("--savedir", type=str, default="/home/work/kby_hgh/audio_lora_teacher_43800_OOD_gpt_prompt_inference", help="결과 비디오 저장 디렉토리")
    parser.add_argument("--bs", type=int, default=2, help="배치 사이즈")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--audio_model_name", type=str, default="auffusion/auffusion-full", 
                        help="사전 학습된 모델 디렉토리 또는 identifier")
    parser.add_argument("--audio_lora_ckpt_path", type=str, default="/home/work/kby_hgh/AUDIO_LORA_CHECKPOINT_0416/1e-6/checkpoint-step-43800/model.safetensors")

    parser.add_argument("--duration", type=float, default=3.2, help="오디오 길이 (초)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale 값")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="DDIM 스케줄러 추론 단계 수")
    parser.add_argument("--eta_audio", type=float, default=0.0, help="오디오 DDIM eta 파라미터")
    args = parser.parse_args()

    # Accelerator 초기화 및 device/dtype 설정
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float32


    # Audio Models
    audio_unet = UNet2DConditionModel.from_pretrained(args.audio_model_name, subfolder="unet")
    audio_unet.eval()
    for param in audio_unet.parameters():
        param.requires_grad = False

    # LoRA config
    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    audio_unet.add_adapter(lora_config)
    audio_unet = load_accelerator_ckpt(audio_unet, args.audio_lora_ckpt_path)
    audio_unet = audio_unet.to(device)

    if not os.path.isdir(args.audio_model_name):
        pretrained_model_name_or_path = snapshot_download(args.audio_model_name)
    else:
        pretrained_model_name_or_path = args.audio_model_name

    # 2-2) VAE 로드
    with torch.no_grad():
        audio_vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae"
        )
    audio_vae = audio_vae.to(device=device,dtype=dtype)
    audio_vae.requires_grad_(False)

    # 2-3) VAE scale factor 기반 ImageProcessor
    audio_vae_scale_factor = 2 ** (len(audio_vae.config.block_out_channels) - 1)
    audio_image_processor = VaeImageProcessor(vae_scale_factor=audio_vae_scale_factor)

    # 2-4) condition_config.json 기반으로 text_encoder_list, tokenizer_list, adapter_list 로딩
    audio_condition_json_path = os.path.join(pretrained_model_name_or_path, "condition_config.json")
    with open(audio_condition_json_path, "r", encoding="utf-8") as f:
        audio_condition_json_list = json.load(f)

    audio_text_encoder_list = []
    audio_tokenizer_list = []
    audio_adapter_list = []

    with torch.no_grad():
        for cond_item in audio_condition_json_list:
            # text encoder / tokenizer
            audio_text_encoder_path = os.path.join(pretrained_model_name_or_path, cond_item["text_encoder_name"])
            audio_tokenizer = AutoTokenizer.from_pretrained(audio_text_encoder_path)
            
            audio_text_encoder_cls = import_model_class_from_model_name_or_path(audio_text_encoder_path)
            audio_text_encoder = audio_text_encoder_cls.from_pretrained(audio_text_encoder_path)

            audio_text_encoder.requires_grad_(False)
            audio_text_encoder = audio_text_encoder.to(device=device, dtype=dtype)

            audio_tokenizer_list.append(audio_tokenizer)
            audio_text_encoder_list.append(audio_text_encoder)

            # condition adapter
            audio_adapter_path = os.path.join(pretrained_model_name_or_path, cond_item["condition_adapter_name"])
            audio_adapter = ConditionAdapter.from_pretrained(audio_adapter_path)
            audio_adapter.requires_grad_(False)
            audio_adapter = audio_adapter.to(device=device,dtype=dtype)

            audio_adapter_list.append(audio_adapter)




    # run_inference 실행
    run_inference(
         accelerator=accelerator,
         unet_model=audio_unet,
         vae=audio_vae,
         image_processor=audio_image_processor,
         text_encoder_list=audio_text_encoder_list,
         adapter_list=audio_adapter_list,
         tokenizer_list=audio_tokenizer_list,
         seed=args.seed,
         prompt_file=args.prompt_file,
         savedir=args.savedir,
         bs=args.bs,
         pretrained_model_name_or_path=args.audio_model_name,
         duration=args.duration,
         guidance_scale=args.guidance_scale,
         num_inference_steps=args.num_inference_steps,
         eta_audio=args.eta_audio
    )

if __name__ == "__main__":
    main()