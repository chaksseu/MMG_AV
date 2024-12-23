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
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip

from accelerate import Accelerator

from scripts.evaluation.funcs import load_model_checkpoint, load_prompts
from utils.utils import instantiate_from_config
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like
from train_MMG_Model_1220_LoRA import CrossModalCoupledUNet
from auffusion_pipe_functions import (
    prepare_extra_step_kwargs, ConditionAdapter, import_model_class_from_model_name_or_path, Generator
)





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


def save_video_with_audio(video_path, audio_path, savedir, base_filename, fps=10):
    # 저장 디렉토리가 존재하지 않으면 생성
    os.makedirs(savedir, exist_ok=True)
    
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    video_with_audio = video_clip.set_audio(audio_clip)
    
    savepath = os.path.join(savedir, f"{base_filename}_combined.mp4")
    
    video_with_audio.write_videofile(
        savepath,
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None
    )

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

def load_cross_modal_unet(
    audio_unet: torch.nn.Module,
    video_unet: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype
) -> CrossModalCoupledUNet:
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device
    }
    cross_modal_model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config).to(device, dtype).eval()
    checkpoint = load_file(checkpoint_path)
    cross_modal_model.load_state_dict(checkpoint)
    return cross_modal_model

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility")
    parser.add_argument("--mode", default="base", type=str, help="Inference mode: {'base'}")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path for video pipeline")
    parser.add_argument("--config", type=str, required=True, help="Config path for video pipeline")
    parser.add_argument("--prompt_file", type=str, required=True, help="Prompt text file")
    parser.add_argument("--savedir", type=str, default="1217_mmg_output_multi_gpus", help="Output directory")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--height", type=int, default=256, help="Video frame height")
    parser.add_argument("--width", type=int, default=256, help="Video frame width")
    parser.add_argument("--frames", type=int, default=40, help="Number of video frames")
    parser.add_argument("--fps", type=float, default=12.5, help="Video fps")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="CFG scale for video")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Pretrained multimodal model path")
    parser.add_argument("--cross_modal_checkpoint_path", type=str, required=True, help="Cross-modal checkpoint path")
    parser.add_argument("--duration", type=float, default=3.2, help="Audio duration in seconds")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Audio guidance scale")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps for video")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta for video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Audio inference steps")
    parser.add_argument("--eta_audio", type=float, default=0.0, help="Eta for audio scheduler")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (handled by accelerate)")
    return parser


@torch.no_grad()
def run_inference(
    args: argparse.Namespace,
    accelerator: Accelerator,
    prompt_sublist: List[str],
    dtype: torch.dtype
):
    try:
        # Set unique seed per process
        unique_seed = args.seed + accelerator.process_index
        seed_everything(unique_seed)
        device = accelerator.device
        generator = torch.Generator(device=device).manual_seed(unique_seed)

        if not prompt_sublist:
            accelerator.print(f"Process {accelerator.process_index}: No prompts to process.")
            return

        audio_length = int(args.duration * 16000)
        latent_time = int(args.fps * args.duration)
        do_audio_cfg = args.guidance_scale > 1.0
        do_video_cfg = args.unconditional_guidance_scale > 1.0

        # Load video pipeline
        config = OmegaConf.load(args.config)
        model_config = config.pop("model", OmegaConf.create())
        video_pipeline = instantiate_from_config(model_config).to(device)
        assert os.path.exists(args.ckpt_path), f"Checkpoint [{args.ckpt_path}] not found!"
        video_pipeline = load_model_checkpoint(video_pipeline, args.ckpt_path)
        video_pipeline.eval()
        video_unet = video_pipeline.model.diffusion_model.to(device, dtype)

        # Model directory
        model_dir = args.pretrained_model_name_or_path
        if not os.path.isdir(model_dir):
            model_dir = snapshot_download(model_dir)

        # Load audio components
        vocoder = load_vocoder(model_dir, device, dtype)
        vae, image_processor = load_vae(model_dir, device, dtype)
        text_encoders, tokenizers, adapters = load_text_encoders(model_dir, device, dtype)
        audio_unet = load_audio_unet(model_dir, device, dtype)
        cross_modal_model = load_cross_modal_unet(audio_unet, video_unet, args.cross_modal_checkpoint_path, device, dtype)

        assert (args.height % 16 == 0) and (args.width % 16 == 0), "Video dimensions must be multiples of 16!"
        latent_h, latent_w = args.height // 8, args.width // 8
        frames = video_pipeline.temporal_length if args.frames < 0 else args.frames
        channels = video_pipeline.channels

        # Create output dirs (main process only)
        if accelerator.is_main_process:
            os.makedirs(args.savedir, exist_ok=True)
            audio_dir = os.path.join(args.savedir, "audio")
            video_dir = os.path.join(args.savedir, "video")
            combined_dir = os.path.join(args.savedir, "combined_video")
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(combined_dir, exist_ok=True)

        accelerator.wait_for_everyone()

        total_prompts = len(prompt_sublist)
        num_batches = (total_prompts + args.bs - 1) // args.bs

        # Initialize DDIM sampler for video
        sampler_state = initialize_ddim_sampler(video_pipeline)
        make_ddim_schedule(sampler_state, ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)

        # Setup progress bar on main process
        if accelerator.is_main_process:
            pbar = tqdm(total=num_batches, desc="Generating", disable=not accelerator.is_main_process)
        else:
            pbar = None

        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.bs
            end_idx = min(start_idx + args.bs, total_prompts)
            current_batch_size = end_idx - start_idx
            current_prompts = prompt_sublist[start_idx:end_idx]

            video_noise_shape = [current_batch_size, channels, frames, latent_h, latent_w]
            fps_tensor = torch.tensor([args.fps] * current_batch_size).to(device).long()

            # Get text embedding for video
            video_text_emb = video_pipeline.get_learned_conditioning(current_prompts)
            if args.mode == 'base':
                video_cond = {"c_crossattn": [video_text_emb], "fps": fps_tensor}
            else:
                raise NotImplementedError("Only 'base' mode is supported.")

            # Unconditional video
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

            # Initialize latents
            video_latents = torch.randn(video_noise_shape, device=device, generator=generator)

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

            extra_step_kwargs = prepare_extra_step_kwargs(audio_scheduler, generator, args.eta_audio)
            timesteps = sampler_state['ddim_timesteps']
            total_steps = timesteps.shape[0]
            time_range = np.flip(timesteps)

            # Denoising loop
            for step_idx, (video_step, audio_step) in enumerate(zip(time_range, audio_timesteps)):
                index = total_steps - step_idx - 1
                video_ts = torch.full((current_batch_size,), video_step, device=device, dtype=torch.long)

                # CFG for audio/video
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

                if do_audio_cfg:
                    audio_out = audio_out_uncond + args.guidance_scale * (audio_out - audio_out_uncond)
                if do_video_cfg:
                    video_out = video_out_uncond + cfg_scale * (video_out - video_out_uncond)

                # Update audio latents
                audio_latents = audio_scheduler.step(audio_out, audio_step, audio_latents, **extra_step_kwargs, return_dict=False)[0]

                # Video DDIM step
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
                noise = sigma_t * torch.randn_like(video_latents) # , generator=generator

                if sampler_state['use_scale']:
                    scale_arr = video_pipeline.scale_arr if use_original_steps else sampler_state['ddim_scale_arr']
                    scale_t = torch.full(size, scale_arr[index], device=device)
                    scale_arr_prev = video_pipeline.scale_arr_prev if use_original_steps else sampler_state['ddim_scale_arr_prev']
                    scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
                    pred_x0 /= scale_t
                    video_latents = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
                else:
                    video_latents = a_prev.sqrt() * pred_x0 + dir_xt + noise

            # Decode audio
            audio_recon_image = vae.decode(audio_latents / vae.config.scaling_factor, return_dict=False)[0]
            do_denormalize = [True] * audio_recon_image.shape[0]
            audio_recon_image = image_processor.postprocess(audio_recon_image, output_type="pt", do_denormalize=do_denormalize)

            audios = []
            for img in audio_recon_image:
                spectrogram = denormalize_spectrogram(img)
                audio_waveform = vocoder.inference(spectrogram, lengths=audio_length)[0]
                audios.append(audio_waveform)

            # Decode video
            video_frames = video_pipeline.decode_first_stage_2DAE(video_latents)

            # Save results (only main process should handle actual file operations)
            # 프로세스별로 구분하려면 파일명에 process_index 추가
            audio_dir = os.path.join(args.savedir, "audio")
            video_dir = os.path.join(args.savedir, "video")
            combined_dir = os.path.join(args.savedir, "combined_video")

            for i, prompt in enumerate(current_prompts):
                safe_prompt = sanitize_filename(prompt)
                sample_index = start_idx + i
                base_filename = f"{safe_prompt}_batch_{sample_index}_proc_{accelerator.process_index}"

                # Save audio
                audio_filepath = os.path.join(audio_dir, f"{base_filename}.wav")
                write(audio_filepath, 16000, audios[i])

                # Save video
                video_filepath = os.path.join(video_dir, f"{base_filename}.mp4")
                single_video_frames = video_frames[i].unsqueeze(0)  
                save_videos(single_video_frames, video_dir, base_filename, fps=args.fps)

                # 오디오가 포함된 비디오 파일 저장
                save_video_with_audio(
                    video_path=video_filepath,
                    audio_path=audio_filepath,
                    savedir=combined_dir,
                    base_filename=base_filename,
                    fps=args.fps
                )

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        accelerator.print(f"Process {accelerator.process_index}: Completed inference. Results saved in {args.savedir}.")

    except Exception as e:
        accelerator.print(f"Process {accelerator.process_index}: Encountered an error.")
        traceback.print_exc()

def main():
    parser = get_parser()
    args = parser.parse_args()

    assert os.path.exists(args.prompt_file), f"Prompt file not found: {args.prompt_file}"
    all_prompts = load_prompts(args.prompt_file)

    # Accelerate 초기화
    #accelerator = Accelerator()
    accelerator = Accelerator(mixed_precision="bf16")

    # 전체 프롬프트를 num_processes에 맞게 균등 분배
    num_processes = accelerator.num_processes
    prompt_subsets = split_prompts_evenly(all_prompts, num_processes)

    # 현재 프로세스 인덱스에 해당하는 프롬프트 subset
    if accelerator.process_index < len(prompt_subsets):
        process_prompts = prompt_subsets[accelerator.process_index]
    else:
        process_prompts = []

    dtype = torch.float32

    # Inference 실행
    run_inference(args, accelerator, process_prompts, dtype)

    # 모든 프로세스 동기화 후 종료
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
