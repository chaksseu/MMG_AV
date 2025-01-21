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
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tqdm import tqdm
import re
import traceback
import contextlib
import sys
import csv

from accelerate import Accelerator
from peft import LoraConfig

from mmg_inference.auffusion_pipe_functions import (
    prepare_extra_step_kwargs, ConditionAdapter, import_model_class_from_model_name_or_path, Generator
)



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
                encoded = text_encoders[idx](input_ids).last_hidden_state

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
        dtype = torch.bfloat16
        
        assert os.path.exists(prompt_file), f"Prompt file not found: {prompt_file}"
        all_prompts = load_prompts(prompt_file)

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
            return

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
        vae, image_processor = load_vae(model_dir, device, dtype)
        text_encoders, tokenizers, adapters = load_text_encoders(model_dir, device, dtype)
        
        audio_unet = unet_model

        # 모델을 Accelerator로 준비
        ###audio_unet = accelerator.prepare(audio_unet)



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

            # Encode audio prompts
            audio_prompt_embeds = encode_audio_prompt(
                text_encoders, tokenizers, adapters,
                max_length=77,
                dtype=dtype,
                prompt=current_prompts,
                device=device,
                do_classifier_free_guidance=do_audio_cfg
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
                    audio_out = audio_out_uncond + .guidance_scale * (audio_out - audio_out_uncond)

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

def main():
    # Accelerate 초기화
    accelerator = Accelerator(mixed_precision="bf16")
    
    # Inference 실행
    run_inference(
        accelerator=accelerator,
        unet_model=unet_model,
        prompt_file=csv_path,
        savedir=inference_path,
        bs=inference_batch_size,
        pretrained_model_name_or_path="auffusion/auffusion-full",
        seed=1234,
        duration=3.2,
        guidance_scale=7.5,
        num_inference_steps=50,
        eta_audio=0.0
        )


if __name__ == '__main__':
    main()
