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

# Custom imports
from auffusion_pipe_functions import (
    prepare_extra_step_kwargs, ConditionAdapter, import_model_class_from_model_name_or_path, Generator
)





def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    if images.min() < 0:
        return (images / 2 + 0.5).clamp(0, 1)
    else:
        return images.clamp(0, 1)     

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
    fps: float = 12.5
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

    # Resolve model directory
    if not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path) 
    model_dir = pretrained_model_name_or_path


    # Load all components

    vocoder = load_vocoder(model_dir, device, dtype)
    vae, image_processor = load_vae(model_dir, device, dtype)
    text_encoder_list, tokenizer_list, adapter_list = load_text_encoders(model_dir, device, dtype)
    audio_unet = load_audio_unet(model_dir, device, dtype)


    # Classifier-free guidance
    do_classifier_free_guidance = guidance_scale > 1.0





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

    # Scheduler setup
    scheduler = PNDMScheduler.from_pretrained(model_dir, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    

    # Example latent shapes (must match model expectation)
    #audio_shape = (batch_size, 4, 32, 40)

    audio_shape = (batch_size, 4, 32, 128)

    latents = randn_tensor(audio_shape, generator=generator, device=device, dtype=audio_text_embed.dtype) * scheduler.init_noise_sigma

    # Extra step kwargs for scheduler
    extra_step_kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)

    # Denoising loop
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = audio_unet(
            latent_model_input,
            t,
            encoder_hidden_states=audio_text_embed,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]



    # Decode audio latents
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    image = image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)


    # Generate audio
    spectrograms, audios = [], []
    for img in image:
        spectrogram = denormalize_spectrogram(img)
        audio = vocoder.inference(spectrogram, lengths=audio_length)[0]
        audios.append(audio)
        spectrograms.append(spectrogram)



    # Save audio output
    audio_savedir = "output/audio"
    os.makedirs(audio_savedir, exist_ok=True)
    for i in range(len(audios)):
        audio_path = os.path.join(audio_savedir, f"audio_{prompt[i]}_batch_idx_{i}.wav")
        audio_path = audio_path.replace(" ", "_")

        audio = audios[i]
        write(audio_path, 16000, audio)


    print("Inference completed.")

############################################################
# Example Usage
############################################################

if __name__ == "__main__":
    # Example call
    # Note: Ensure that all paths, model configs, and shapes are adapted to your setup.
    with torch.autocast("cuda"):
        run_inference(
            prompt=["hammering nails", "dog barking", "lions roaring", "people sneezing"],
            #, "skateboarding", "hammering nails", "dog barking", "playing badminton", "chopping wood", "lions roaring", "people sneezing", "people eating crisps"
            pretrained_model_name_or_path="auffusion/auffusion",
            cross_modal_checkpoint_path="mmg_checkpoints/lr_1e-06_batch_1536_global_step_1200_vggsound_sparse/model.safetensors",
            device="cuda:0",
            dtype=torch.float16,
            seed=42,
            guidance_scale=1.0,
            num_train_timesteps=1000,
            num_inference_steps=100,
            duration=10.0,
            eta=0.0,
            output_type="pt",
            fps = 16
        )
