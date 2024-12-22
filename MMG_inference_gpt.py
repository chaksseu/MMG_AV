import os
import json
from typing import List, Optional

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.io.wavfile import write

from transformers import AutoTokenizer
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDIMScheduler,
    PNDMScheduler
)
from diffusers.image_processor import VaeImageProcessor

from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

# Custom imports
from train_MMG_Model_1216 import CrossModalCoupledUNet
from auffusion_pipe_functions import (
    prepare_extra_step_kwargs, ConditionAdapter, import_model_class_from_model_name_or_path, Generator
)
from utils.utils import instantiate_from_config
from safetensors.torch import load_file
from tqdm import tqdm

import soundfile as sf
from moviepy.editor import ImageSequenceClip, AudioFileClip


    

############################################################
# Prompt Encoding
############################################################

def encode_audio_prompt(
    text_encoder_list,
    tokenizer_list,
    adapter_list,
    tokenizer_model_max_length,
    dtype,
    prompt,
    device,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    do_classifier_free_guidance=False
):
    """
    Encode textual prompts for the audio modality using multiple text encoders and adapters.
    """
    assert len(text_encoder_list) == len(tokenizer_list), "Mismatched text_encoder and tokenizer counts."
    if adapter_list is not None:
        assert len(text_encoder_list) == len(adapter_list), "Mismatched text_encoder and adapter counts."

    def get_prompt_embeds(prompt_list, device):
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        prompt_embeds_list = []
        for p in prompt_list:
            encoder_hidden_states_list = []
            for j in range(len(text_encoder_list)):
                input_ids = tokenizer_list[j](p, return_tensors="pt").input_ids.to(device)
                cond_embs = text_encoder_list[j](input_ids).last_hidden_state
                # Pad/truncate embeddings
                if cond_embs.shape[1] < tokenizer_model_max_length:
                    pad_len = tokenizer_model_max_length - cond_embs.shape[1]
                    cond_embs = F.pad(cond_embs, (0, 0, 0, pad_len), value=0)
                else:
                    cond_embs = cond_embs[:, :tokenizer_model_max_length, :]

                if adapter_list is not None:
                    cond_embs = adapter_list[j](cond_embs)
                    encoder_hidden_states_list.append(cond_embs)

            prompt_embeds_batch = torch.cat(encoder_hidden_states_list, dim=1)
            prompt_embeds_list.append(prompt_embeds_batch)

        return torch.cat(prompt_embeds_list, dim=0)

    if prompt_embeds is None:           
        prompt_embeds = get_prompt_embeds(prompt, device)

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if do_classifier_free_guidance:
        # Create negative prompt embeddings for classifier-free guidance
        negative_prompt_embeds = torch.zeros_like(prompt_embeds, dtype=prompt_embeds.dtype, device=device)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds

############################################################
# Spectrogram Utility
############################################################

def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    if images.min() < 0:
        return (images / 2 + 0.5).clamp(0, 1)
    else:
        return images.clamp(0, 1)     
    
def denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1, 
) -> torch.Tensor:
    
    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))

    max_value = np.log(max_value)
    min_value = np.log(min_value)
    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])    
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)        
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]
    # Reverse the power curve
    data = torch.pow(data, 1 / power)
    # Rescale to max value
    spectrogram = data * (max_value - min_value) + min_value

    return spectrogram

############################################################
# Video Saving
############################################################

def save_videos(batch_tensors, savedir, filenames, fps=12.5):
    """
    Save a batch of video tensors as mp4 files.
    Input: [B, C, T, H, W]
    """
    os.makedirs(savedir, exist_ok=True)
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        # Rearrange to [T, C, H, W]
        video = video.permute(1, 0, 2, 3)
        frame_grids = [torchvision.utils.make_grid(f.unsqueeze(0), nrow=1) for f in video]
        grid = torch.stack(frame_grids, dim=0)  # [T, 3, H, W]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

############################################################
# Model and Pipeline Initialization
############################################################


def load_vocoder(model_path: str, device: str, dtype: torch.dtype):
    """
    Load the vocoder model from a given path.
    """
    return Generator.from_pretrained(model_path, subfolder="vocoder").to(device, dtype)

def load_vae(model_dir: str, device: str, dtype: torch.dtype):
    """
    Load the VAE model for decoding latent representations.
    """
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae").to(device, dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    return vae, image_processor

def load_text_encoders(model_dir: str, device: str, dtype: torch.dtype):
    """
    Load text encoders, tokenizers, and adapters based on condition config JSON.
    """
    condition_json_path = os.path.join(model_dir, "condition_config.json")
    condition_json_list = json.loads(open(condition_json_path).read())
    text_encoder_list, tokenizer_list, adapter_list = [], [], []

    with torch.no_grad():
        for condition_item in condition_json_list:
            text_encoder_path = os.path.join(model_dir, condition_item["text_encoder_name"])
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            tokenizer_list.append(tokenizer)

            text_encoder_cls = import_model_class_from_model_name_or_path(text_encoder_path)
            text_encoder = text_encoder_cls.from_pretrained(text_encoder_path).to(device, dtype)
            text_encoder_list.append(text_encoder)

            adapter_path = os.path.join(model_dir, condition_item["condition_adapter_name"])
            adapter = ConditionAdapter.from_pretrained(adapter_path).to(device, dtype)
            adapter_list.append(adapter)

    return text_encoder_list, tokenizer_list, adapter_list

def load_audio_unet(model_dir: str, device, dtype):
    """
    Load the Audio UNet model.
    """
    return UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet").to(device, dtype)


def load_video_unet(device, dtype):
    """
    Load the Video UNet model based on provided configs.
    Adjust the path and configs as per your setup.
    """
    video_config = OmegaConf.load('configs/inference_t2v_512_v2.0.yaml')
    video_model = instantiate_from_config(video_config.model).to(device, dtype)
    state_dict = torch.load('scripts/evaluation/model.ckpt')['state_dict']
    video_model.load_state_dict(state_dict, strict=True)
    video_unet = video_model.model.diffusion_model.to(device, dtype)
    return video_model.eval(), video_unet.eval()


def load_cross_modal_unet(audio_unet: nn.Module, video_unet: nn.Module, checkpoint_path: str, device: str, dtype: torch.dtype) -> CrossModalCoupledUNet:
    """
    Initialize the combined CrossModalCoupledUNet model and load pretrained weights.

    Args:
        audio_unet (nn.Module): Pretrained Audio UNet model.
        video_unet (nn.Module): Pretrained Video UNet model.
        checkpoint_path (str): Path to the saved CrossModalCoupledUNet checkpoint.
        device (str): Device to load the model on.
        dtype (torch.dtype): Data type for the model.

    Returns:
        CrossModalCoupledUNet: Loaded cross-modal model.
    """
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device
    }
    model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config).to(device, dtype).eval()
    
    checkpoint = load_file(checkpoint_path) 
    model.load_state_dict(checkpoint)

    model.to(device, dtype)
    return model

############################################################
# Inference Routine
############################################################
@torch.no_grad()
def run_inference(
    prompt: List[str],
    pretrained_model_name_or_path: str,
    cross_modal_checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 25,
    duration: float = 3.2,
    eta: float = 0.0,
    output_type: str = "pt",
    fps: float = 12.5,
    save_dir: str = "output"
):
    """
    Run the inference process to generate both audio and video from text prompts.

    Args:
        prompt (List[str]): The text prompt(s).
        pretrained_model_name_or_path (str): Path or name of the pretrained model directory.
        device (str): Compute device.
        dtype (torch.dtype): Precision type (e.g. torch.float32).
        seed (int): Random seed for reproducibility.
        guidance_scale (float): Scale for classifier-free guidance.
        num_train_timesteps (int): Total train steps in the scheduler.
        num_inference_steps (int): Number of inference steps to run.
        duration (float): Duration of the generated audio in seconds.
        eta (float): DDIM scheduler parameter.
        output_type (str): Output data type for the image processor.
    """


    # Set random seed and device
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    audio_length = int(duration * 16000)
    batch_size = len(prompt)
    latent_time = int(fps * duration)


    # Resolve model directory
    if not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path) 
    model_dir = pretrained_model_name_or_path

    # Load all components
    vocoder = load_vocoder(model_dir, device, dtype)
    vae, image_processor = load_vae(model_dir, device, dtype)
    text_encoder_list, tokenizer_list, adapter_list = load_text_encoders(model_dir, device, dtype)
    audio_unet = load_audio_unet(model_dir, device, dtype)
    video_model, video_unet = load_video_unet(device, dtype)

    model = load_cross_modal_unet(audio_unet, video_unet, cross_modal_checkpoint_path, device, dtype)

    # Classifier-free guidance
    do_classifier_free_guidance = guidance_scale > 1.0

    # Scheduler setup
    scheduler_a = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
    scheduler_v = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")

    scheduler_a.set_timesteps(num_inference_steps, device=device)
    scheduler_v.set_timesteps(num_inference_steps, device=device)

    timesteps = scheduler_a.timesteps
    
    # fps
    fps = torch.tensor([fps]*batch_size).long().to(device)


    # Encode input prompt for audio and video
    audio_text_embed = encode_audio_prompt(
        text_encoder_list=text_encoder_list,
        tokenizer_list=tokenizer_list,
        adapter_list=adapter_list,
        tokenizer_model_max_length=77,
        dtype=dtype,
        prompt=prompt,
        device=device,
        do_classifier_free_guidance=do_classifier_free_guidance
    )

    video_text_embed = video_model.get_learned_conditioning(prompt)
    video_text_embed = video_text_embed.to(device)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(video_text_embed, dtype=video_text_embed.dtype, device=device)
        video_text_embed = torch.cat([negative_prompt_embeds, video_text_embed])


    # Example latent shapes (must match model expectation)
    audio_shape = (batch_size, 4, 32, latent_time)
    video_shape = (batch_size, 4, latent_time, 32, 32)

    #audio_shape = (batch_size, 4, 32, 128)
    #video_shape = (batch_size, 4, 32, 40, 64)

    audio_latents = randn_tensor(audio_shape, generator=generator, device=device, dtype=audio_text_embed.dtype) * scheduler_a.init_noise_sigma
    video_latents = randn_tensor(video_shape, generator=generator, device=device, dtype=video_text_embed.dtype) * scheduler_v.init_noise_sigma

    # Extra step kwargs for scheduler
    extra_step_kwargs = prepare_extra_step_kwargs(scheduler_a, generator, eta)


    # Denoising loop
    for i, t in enumerate(timesteps):
        _t = t
        t = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)

        if do_classifier_free_guidance:
            # Duplicate latents for guidance
            audio_input = torch.cat([audio_latents] * 2)
            video_input = torch.cat([video_latents] * 2)
            t = torch.cat([t] * 2)
        else:
            audio_input = audio_latents
            video_input = video_latents

        audio_input = scheduler_a.scale_model_input(audio_input, _t)
        video_input = scheduler_v.scale_model_input(video_input, _t)


        
        with torch.no_grad():
            #video_output = video_input      
            audio_output, video_output = model(
                audio_latents=audio_input,
                audio_timestep=_t,
                audio_encoder_hidden_states=audio_text_embed,
                video_latents=video_input,
                video_timestep=t,
                video_context=video_text_embed,
                video_fps=fps
            )

        # Classifier-free guidance
        if do_classifier_free_guidance:
            audio_output_uncond, audio_output_text = audio_output.chunk(2)
            audio_output = audio_output_uncond + guidance_scale * (audio_output_text - audio_output_uncond)
            video_output_uncond, video_output_text = video_output.chunk(2)
            video_output = video_output_uncond + guidance_scale * (video_output_text - video_output_uncond)

        # Scheduler step
        audio_latents = scheduler_a.step(audio_output, _t, audio_latents, **extra_step_kwargs, return_dict=False)[0]
        video_latents = scheduler_v.step(video_output, _t, video_latents, **extra_step_kwargs, return_dict=False)[0] # 스케쥴러가 달라서 그런가 에러뜸
        


    # Decode audio latents
    audio_image = vae.decode(audio_latents / vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * audio_image.shape[0]
    audio_image = image_processor.postprocess(audio_image, output_type=output_type, do_denormalize=do_denormalize)

    # Convert spectrograms to audio
    spectrograms, audios = [], []
    for img in audio_image:
        spectrogram = denormalize_spectrogram(img)
        audio = vocoder.inference(spectrogram, lengths=audio_length)[0]
        audios.append(audio)
        spectrograms.append(spectrogram)


    # Save audio output
    audio_savedir = os.path.join(save_dir, "audio")
    os.makedirs(audio_savedir, exist_ok=True)
    for i in range(len(audios)):
        audio_path = os.path.join(audio_savedir, f"audio_{prompt[i]}_batch_idx_{i}.wav")
        audio_path = audio_path.replace(" ", "_")
        audio = audios[i]
        write(audio_path, 16000, audio)


    # Decode video latents (assuming a custom decode function is available)
    with torch.no_grad():
        batch_images = video_model.decode_first_stage_2DAE(video_latents)
        video_savedir = os.path.join(save_dir, "video")
        os.makedirs(video_savedir, exist_ok=True)
        video_filenames = [f"video_{prompt[i]}_batch_idx_{i}" for i in range(batch_size)]
        save_videos(batch_images, video_savedir, video_filenames, fps=12.5)

    print("Inference completed.")










############################################################
# Example Usage
############################################################

if __name__ == "__main__":
    # Example call
    # Note: Ensure that all paths, model configs, and shapes are adapted to your setup.

    device = "cuda:7"

    with torch.autocast(device):
        run_inference(
            prompt=["people sneezing", "dog barking", "hammering nails", "lions roaring"],
            #, "skateboarding", "hammering nails", "dog barking", "playing badminton", "chopping wood", "lions roaring", "people sneezing", "people eating crisps"
            pretrained_model_name_or_path="auffusion/auffusion",
            cross_modal_checkpoint_path="mmg_checkpoints/1216_linear_lr_1e-06_batch_1024_global_step_2000_vggsound_sparse/model.safetensors",
            device=device,
            dtype=torch.float16,
            seed=42,
            guidance_scale=1.0,
            num_train_timesteps=1000,
            num_inference_steps=100,
            duration=10.0,
            eta=0.0,
            output_type="pt",
            fps = 16,
            save_dir= "1216_output_02"
        )
