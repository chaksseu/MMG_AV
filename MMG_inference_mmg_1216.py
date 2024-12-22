import argparse
import datetime
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from scipy.io.wavfile import write
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tqdm import tqdm
import torchvision
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import re


# Custom imports
from scripts.evaluation.funcs import load_model_checkpoint, load_prompts, save_videos
from utils.utils import instantiate_from_config
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like
from train_MMG_Model_1216 import CrossModalCoupledUNet
from auffusion_pipe_functions import (
    prepare_extra_step_kwargs, ConditionAdapter, import_model_class_from_model_name_or_path, Generator
)






############################################################
# Audio Prompt Encoding
############################################################

def encode_audio_prompt(
    text_encoders: list,
    tokenizers: list,
    adapters: list,
    max_length: int,
    dtype,
    prompt: list,
    device,
    do_classifier_free_guidance=False,
    prompt_embeds: Optional[torch.FloatTensor] = None
):
    """
    Encode textual prompts for the audio modality using multiple text encoders and adapters.
    """
    assert len(text_encoders) == len(tokenizers), "Mismatched text_encoder and tokenizer counts."
    if adapters is not None:
        assert len(text_encoders) == len(adapters), "Mismatched text_encoder and adapter counts."

    def get_prompt_embeds(prompt_list, device):
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        embeds_all = []
        for p in prompt_list:
            cond_embs_list = []
            for idx in range(len(text_encoders)):
                input_ids = tokenizers[idx](p, return_tensors="pt").input_ids.to(device)
                encoded = text_encoders[idx](input_ids).last_hidden_state

                # Pad/truncate embeddings to max_length
                if encoded.shape[1] < max_length:
                    pad_len = max_length - encoded.shape[1]
                    encoded = F.pad(encoded, (0, 0, 0, pad_len), value=0)
                else:
                    encoded = encoded[:, :max_length, :]

                # Condition adapter
                if adapters is not None:
                    encoded = adapters[idx](encoded)
                cond_embs_list.append(encoded)

            merged_embs = torch.cat(cond_embs_list, dim=1)
            embeds_all.append(merged_embs)
        return torch.cat(embeds_all, dim=0)

    if prompt_embeds is None:
        prompt_embeds = get_prompt_embeds(prompt, device)

    return prompt_embeds.to(dtype=dtype, device=device)


############################################################
# Spectrogram Utility
############################################################

def denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1
) -> torch.Tensor:
    """
    Convert normalized spectrogram back to original scale.
    """
    assert len(data.shape) == 3, f"Expected 3D tensor, got shape {data.shape}"

    max_value = np.log(max_value)
    min_value = np.log(min_value)

    # Flip Y-axis
    data = torch.flip(data, [1])

    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    assert data.shape[0] == 3, "Spectrogram must have 3 channels"

    data = data[0]
    data = torch.pow(data, 1 / power)
    spectrogram = data * (max_value - min_value) + min_value
    return spectrogram


############################################################
# DDIM Utilities
############################################################

def initialize_ddim_sampler(model, schedule="linear"):
    """
    Initialize DDIM sampler state.
    """
    return {
        "model": model,
        "ddpm_num_timesteps": model.num_timesteps,
        "schedule": schedule,
        "counter": 0,
        "use_scale": model.use_scale
    }

def register_buffer(sampler_state, name, attr):
    if isinstance(attr, torch.Tensor) and attr.device != torch.device("cuda"):
        attr = attr.to(torch.device("cuda"))
    sampler_state[name] = attr

def make_ddim_schedule(sampler_state, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
    """
    Prepare DDIM schedules.
    """
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


############################################################
# Model and Pipeline Initialization
############################################################

def load_vocoder(model_path: str, device: str, dtype: torch.dtype):
    """Load the vocoder model."""
    return Generator.from_pretrained(model_path, subfolder="vocoder").to(device, dtype)

def load_vae(model_dir: str, device: str, dtype: torch.dtype):
    """Load VAE model and image processor."""
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae").to(device, dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    return vae, image_processor

def load_text_encoders(model_dir: str, device: str, dtype: torch.dtype):
    """Load text encoders and adapters."""
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

def load_audio_unet(model_dir: str, device, dtype):
    """Load Audio UNet."""
    return UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet").to(device, dtype)

def load_cross_modal_unet(
    audio_unet: torch.nn.Module, 
    video_unet: torch.nn.Module, 
    checkpoint_path: str, 
    device: str, 
    dtype: torch.dtype
) -> CrossModalCoupledUNet:
    """Initialize and load CrossModalCoupledUNet."""
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device
    }
    cross_modal_model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config).to(device, dtype).eval()
    checkpoint = load_file(checkpoint_path)
    cross_modal_model.load_state_dict(checkpoint)
    return cross_modal_model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility")
    parser.add_argument("--mode", default="base", type=str, help="Inference mode: {'base'}")
    parser.add_argument("--ckpt_path", type=str, default='scripts/evaluation/model.ckpt', help="Checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="Config path")
    parser.add_argument("--prompt_file", type=str, default="prompts/test_prompts.txt", help="Prompt text file")
    parser.add_argument("--savedir", type=str, default="1217_mmg_output_05", help="Output directory")
    parser.add_argument("--savefps", type=float, default=12.5, help="FPS for saved video")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="DDIM eta")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--height", type=int, default=256, help="Video frame height")
    parser.add_argument("--width", type=int, default=256, help="Video frame width")
    parser.add_argument("--frames", type=int, default=40, help="Number of video frames")
    parser.add_argument("--fps", type=float, default=12.5, help="Video fps")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="CFG scale")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="auffusion/auffusion", help="Pretrained model path or HF Hub name")
    parser.add_argument("--cross_modal_checkpoint_path", type=str, default="mmg_checkpoints/1216_new_lr_1e-06_batch_1024_global_step_1000_vggsound_sparse/model.safetensors", help="Cross-modal checkpoint path")
    parser.add_argument("--duration", type=float, default=3.2, help="Audio duration in seconds")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Audio guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Audio scheduler inference steps")
    parser.add_argument("--eta_audio", type=float, default=1.0, help="Eta for audio scheduler")
    parser.add_argument("--device_id", type=int, default=6, help="GPU device ID")
    return parser


@torch.no_grad()
def run_inference(args, device, dtype):
    # Set seed and device
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device_id}")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    audio_length = int(args.duration * 16000)
    latent_time = int(args.fps * args.duration)
    do_audio_cfg = args.guidance_scale > 1.0
    do_video_cfg = args.unconditional_guidance_scale > 1.0

    # Load prompts
    if os.path.exists(args.prompt_file):
        prompt_list = load_prompts(args.prompt_file)
    else:
        prompt_list = ["people sneezing", "dog barking", "hammering nails", "lions roaring"]

    # Load config and video model
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    video_pipeline = instantiate_from_config(model_config).to(device)
    assert os.path.exists(args.ckpt_path), f"Checkpoint [{args.ckpt_path}] not found!"
    video_pipeline = load_model_checkpoint(video_pipeline, args.ckpt_path)
    video_pipeline.eval()
    video_unet = video_pipeline.model.diffusion_model.to(device, dtype)

    # Resolve model directory
    model_dir = args.pretrained_model_name_or_path
    if not os.path.isdir(model_dir):
        model_dir = snapshot_download(model_dir)

    # Load components for audio
    vocoder = load_vocoder(model_dir, device, dtype)
    vae, image_processor = load_vae(model_dir, device, dtype)
    text_encoders, tokenizers, adapters = load_text_encoders(model_dir, device, dtype)
    audio_unet = load_audio_unet(model_dir, device, dtype)
    cross_modal_model = load_cross_modal_unet(audio_unet, video_unet, args.cross_modal_checkpoint_path, device, dtype)

    # Validate video size
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Video dimensions must be multiples of 16!"
    latent_h, latent_w = args.height // 8, args.width // 8
    frames = video_pipeline.temporal_length if args.frames < 0 else args.frames
    channels = video_pipeline.channels

    # Prepare output directories
    os.makedirs(args.savedir, exist_ok=True)

    start = time.time()
    total_prompts = len(prompt_list)
    num_batches = (total_prompts + args.bs - 1) // args.bs

    # Initialize DDIM sampler for video
    sampler_state = initialize_ddim_sampler(video_pipeline)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.bs
        end_idx = min(start_idx + args.bs, total_prompts)
        current_batch_size = end_idx - start_idx
        current_prompts = prompt_list[start_idx:end_idx]

        video_noise_shape = [current_batch_size, channels, frames, latent_h, latent_w]
        fps_tensor = torch.tensor([args.fps] * current_batch_size).to(device).long()

        # Video text embedding
        video_text_emb = video_pipeline.get_learned_conditioning(current_prompts)
        if args.mode == 'base':
            video_cond = {"c_crossattn": [video_text_emb], "fps": fps_tensor}
        else:
            raise NotImplementedError("Only 'base' mode is supported.")

        # Unconditional video guidance
        cfg_scale = args.unconditional_guidance_scale
        uncond_type = video_pipeline.uncond_type
        if cfg_scale != 1.0:
            if uncond_type == "empty_seq":
                uncond_video_emb = video_pipeline.get_learned_conditioning(current_batch_size * [""])
            elif uncond_type == "zero_embed":
                c_emb = video_cond["c_crossattn"][0]
                uncond_video_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError("Unknown uncond_type")

            uncond_video_cond = {k: video_cond[k] for k in video_cond.keys()}
            uncond_video_cond.update({'c_crossattn': [uncond_video_emb]})
        else:
            uncond_video_cond = None

        # Make DDIM schedule for video
        make_ddim_schedule(sampler_state, ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)

        # Initialize video latents
        video_latents = torch.randn(video_noise_shape, device=device)
        timesteps = sampler_state['ddim_timesteps']
        total_steps = timesteps.shape[0]
        time_range = np.flip(timesteps)

        # Audio scheduler
        audio_scheduler = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        audio_scheduler.set_timesteps(args.num_inference_steps, device=device)
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

        # Extra kwargs for audio scheduler
        extra_step_kwargs = prepare_extra_step_kwargs(audio_scheduler, generator, args.eta_audio)

        # Denoising loop
        for step_idx, (video_step, audio_step) in enumerate(tqdm(zip(time_range, audio_timesteps),
                                                                 total=total_steps, desc="Denoising steps")):
            index = total_steps - step_idx - 1
            video_ts = torch.full((current_batch_size,), video_step, device=device, dtype=torch.long)

            # Audio & Video CFG
            if do_audio_cfg or do_video_cfg:
                neg_audio_prompt_embeds = torch.zeros_like(audio_prompt_embeds, dtype=audio_prompt_embeds.dtype, device=device)
                audio_uncond_input = audio_scheduler.scale_model_input(audio_latents, audio_step)
                audio_out_uncond, video_out_uncond = cross_modal_model(
                    audio_latents=audio_uncond_input,
                    audio_timestep=audio_step,
                    audio_encoder_hidden_states=neg_audio_prompt_embeds,
                    video_latents=video_latents,
                    video_timestep=video_ts,
                    video_context=uncond_video_cond['c_crossattn'][0] if uncond_video_cond else None,
                    video_fps=uncond_video_cond['fps'] if uncond_video_cond else None
                )

            # Audio forward pass
            audio_input = audio_scheduler.scale_model_input(audio_latents, audio_step)
            audio_out, video_out = cross_modal_model(
                audio_latents=audio_input,
                audio_timestep=audio_step,
                audio_encoder_hidden_states=audio_prompt_embeds,
                video_latents=video_latents,
                video_timestep=video_ts,
                video_context=video_cond['c_crossattn'][0],
                video_fps=video_cond['fps']
            )

            # Merge guided outputs for audio and video
            if do_audio_cfg:
                audio_out = audio_out_uncond + args.guidance_scale * (audio_out - audio_out_uncond)
            if do_video_cfg:
                video_out = video_out_uncond + cfg_scale * (video_out - video_out_uncond)

            # Update audio latents
            audio_latents = audio_scheduler.step(audio_out, audio_step, audio_latents, **extra_step_kwargs, return_dict=False)[0]

            # Video DDIM update
            use_original_steps = False
            alphas = video_pipeline.alphas_cumprod if use_original_steps else sampler_state['ddim_alphas']
            alphas_prev = video_pipeline.alphas_cumprod_prev if use_original_steps else sampler_state['ddim_alphas_prev']
            sqrt_oma = video_pipeline.sqrt_one_minus_alphas_cumprod if use_original_steps else sampler_state['ddim_sqrt_one_minus_alphas']
            sigmas = sampler_state['ddim_sigmas_for_original_num_steps'] if use_original_steps else sampler_state['ddim_sigmas']

            is_video = (video_latents.dim() == 5)
            size = (current_batch_size, 1, 1, 1, 1) if is_video else (current_batch_size, 1, 1, 1)
            a_t = torch.full(size, alphas[index], device=device)
            a_prev = torch.full(size, alphas_prev[index], device=device)
            sigma_t = torch.full(size, sigmas[index], device=device)
            sqrt_oma_t = torch.full(size, sqrt_oma[index], device=device)

            pred_x0 = (video_latents - sqrt_oma_t * video_out) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * video_out
            noise = sigma_t * noise_like(video_latents.shape, device, False)

            if sampler_state['use_scale']:
                scale_arr = video_pipeline.scale_arr if use_original_steps else sampler_state['ddim_scale_arr']
                scale_t = torch.full(size, scale_arr[index], device=device)
                scale_arr_prev = video_pipeline.scale_arr_prev if use_original_steps else sampler_state['ddim_scale_arr_prev']
                scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
                pred_x0 /= scale_t
                video_latents = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
            else:
                video_latents = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # Decode audio from latents
        audio_recon_image = vae.decode(audio_latents / vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * audio_recon_image.shape[0]
        audio_recon_image = image_processor.postprocess(audio_recon_image, output_type="pt", do_denormalize=do_denormalize)

        # Convert spectrograms to audio
        audios = []
        for img in audio_recon_image:
            spectrogram = denormalize_spectrogram(img)
            audio_waveform = vocoder.inference(spectrogram, lengths=audio_length)[0]
            audios.append(audio_waveform)

        # Save audio files
        audio_dir = os.path.join(args.savedir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        for i, p_text in enumerate(current_prompts):
            audio_filename = f"audio_{p_text.replace(' ', '_')}_batch_{i}.wav"
            write(os.path.join(audio_dir, audio_filename), 16000, audios[i])

        # Decode video from latents
        video_dir = os.path.join(args.savedir, "video")
        os.makedirs(video_dir, exist_ok=True)
        video_frames = video_pipeline.decode_first_stage_2DAE(video_latents)
        video_samples = video_frames.unsqueeze(1)
        save_videos(video_samples, video_dir, current_prompts, fps=args.savefps)


    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference:", now)
    parser = get_parser()
    args = parser.parse_args()
    dtype = torch.float32
    run_inference(args, device='cuda', dtype=dtype)



