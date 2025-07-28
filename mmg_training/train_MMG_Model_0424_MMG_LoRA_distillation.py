import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
import json
import random
from datetime import timedelta
import csv
import logging

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

from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock,
    SpatialTransformer,
    TemporalTransformer,
    CrossModalTransformer,
    Downsample,
    Upsample,
    TimestepBlock
)

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


from dataset_mmg_distillation_0424 import AudioVideoDataset
from mmg_inference.auffusion_pipe_functions_copy_0123 import (
    encode_audio_prompt,
    ConditionAdapter,
    import_model_class_from_model_name_or_path,
    retrieve_latents
)
from MMG_multi_gpu_inference_mmg_0325 import run_inference
from run_audio_eval import evaluate_audio_metrics
from run_video_eval import evaluate_video_metrics
from run_imagebind_score import evaluate_imagebind_score
from run_av_align import evaluate_av_align_score


from torch.utils.tensorboard import SummaryWriter  # <--- TensorBoard SummaryWriter 추가



os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "0"



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--duration', type=float, default=3.2, help='Duration of the process in seconds.')
    parser.add_argument('--videocrafter_config', type=str, default='configs/inference_t2v_512_v2.0.yaml', help='Path to the videocrafter configuration file.')
    parser.add_argument('--videocrafter_ckpt_path', type=str, default='scripts/evaluation/model.ckpt', help='Path to the videocrafter checkpoint file.')
    parser.add_argument('--audio_model_name', type=str, default='auffusion/auffusion-full', help='Name of the audio model.')
    parser.add_argument('--height', type=int, default=320, help='Height of the video in pixels.')
    parser.add_argument('--width', type=int, default=512, help='Width of the video in pixels.')
    parser.add_argument('--frames', type=int, default=40, help='Number of frames in the video.')
    parser.add_argument('--fps', type=float, default=12.5, help='Frames per second for the video.')
    parser.add_argument('--audio_loss_weight', type=float, default=1.0, help='Loss weight for the TAV audio component.')
    parser.add_argument('--video_loss_weight', type=float, default=4.0, help='Loss weight for the TAV video component.')
    parser.add_argument('--ta_audio_loss_weight', type=float, default=1.0, help='Loss weight for the TA audio component.')
    parser.add_argument('--tv_video_loss_weight', type=float, default=4.0, help='Loss weight for the TV video component.')
    
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='Number of steps to accumulate gradients.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training.')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size.')
    parser.add_argument('--num_epochs', type=int, default=128, help='Number of training epochs.')
    parser.add_argument('--csv_path', type=str, default='/workspace/processed_vggsound_sparse_0218/processed_vggsound_sparse_mmg.csv', help='Path to the CSV file containing data information.')
    parser.add_argument('--spectrogram_dir', type=str, default='/workspace/processed_vggsound_sparse_0218/spec', help='Directory where spectrogram files are stored.')
    parser.add_argument('--video_dir', type=str, default='/workspace/processed_vggsound_sparse_0218/video', help='Directory where video files are stored.')
    
    parser.add_argument('--ta_csv_path', type=str, default="/workspace/data/MMG_TA_dataset_audiocaps_wavcaps/MMG_TA_dataset_filtered_0321.csv", help='Path to the CSV file containing data information.')
    parser.add_argument('--ta_spectrogram_dir', type=str, default="/workspace/data/MMG_TA_dataset_audiocaps_wavcaps_spec_0320", help='Directory where spectrogram files are stored.')
    parser.add_argument('--tv_csv_path', type=str, default="/workspace/processed_OpenVid_0321.csv"  , help='Path to the CSV file containing data information.')
    parser.add_argument('--tv_video_dir', type=str, default="/workspace/data/preprocessed_openvid_videos_train_0318", help='Directory where video files are stored.')

    parser.add_argument('--sampling_rate', type=int, default=16000, help='Audio sampling rate.')
    parser.add_argument('--hop_size', type=int, default=160, help='Hop size for spectrogram calculation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--date', type=str, default='0224', help='Experiment date.')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
    parser.add_argument('--dtype', type=str, default='bf16', help='data type (mixed_precision)')
    parser.add_argument('--cross_modal_checkpoint_path', type=str, default=None, help='Path to the cross-modal checkpoint file.')
    parser.add_argument('--video_lora_ckpt_path', type=str, default='/workspace/video_lora_training_checkpoints_0213/checkpoint-step-16384/model.safetensors', help='video_lora_ckpt_path')
    parser.add_argument('--audio_lora_ckpt_path', type=str, default='/workspace/GCP_BACKUP_0213/checkpoint-step-6400/model.safetensors', help='audio_lora_ckpt_path')

    parser.add_argument('--inference_save_path', type=str, default='/workspace/MMG_SAVE_FOLDER', help='Directory where outputs will be saved.')
    parser.add_argument('--ckpt_save_path', type=str, default='/workspace/MMG_CHECKPOINT', help='Directory where outputs will be saved.')
    parser.add_argument('--inference_batch_size', type=int, default=1, help='Inference batch size.')
    parser.add_argument('--audio_ddim_eta', type=float, default=0.0, help='DDIM eta parameter for audio generation.')
    parser.add_argument('--video_ddim_eta', type=float, default=0.0, help='DDIM eta parameter for video generation.')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps to perform.')
    parser.add_argument('--audio_guidance_scale', type=float, default=7.5, help='Scale factor for audio guidance.')
    parser.add_argument('--video_unconditional_guidance_scale', type=float, default=12.0, help='Scale factor for video unconditional guidance.')
    parser.add_argument('--eval_every', type=int, default=1, help='eval & save ckpt step')
    
    parser.add_argument('--vgg_csv_path', type=str, default='/workspace/vggsound_sparse_curated_292.csv', help='Directory where outputs will be saved.')
    parser.add_argument('--vgg_gt_test_path', type=str, default='/workspace/vggsound_sparse_test_curated_final', help='Directory where outputs will be saved.')
    parser.add_argument('--ac_csv_path', type=str, default='/workspace/processed_vggsound_sparse_0218/avsync_test', help='Directory where outputs will be saved.')
    parser.add_argument('--ac_gt_test_path', type=str, default='/workspace/processed_vggsound_sparse_0218/avsync_gt_test.csv', help='Directory where outputs will be saved.')
    parser.add_argument('--vbench_csv_path', type=str, default='/workspace/processed_vggsound_sparse_0218/avsync_test', help='Directory where outputs will be saved.')
    parser.add_argument('--vbench_gt_test_path', type=str, default='/workspace/processed_vggsound_sparse_0218/avsync_gt_test.csv', help='Directory where outputs will be saved.')

    parser.add_argument('--tensorboard_log_dir', type=str, default='runs', help='TensorBoard log directory.')
    parser.add_argument('--infer_name', type=str, default='infer_name', help='TensorBoard log directory.')


    args = parser.parse_args()
    return args


def load_accelerator_ckpt(model: torch.nn.Module, checkpoint_path: str):
    checkpoint = load_file(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model

def split_prompts_evenly(prompts, num_splits):
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

def load_prompts(prompt_file: str):
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


def evaluate_model(args, accelerator, target_csv_files, eval_id, target_path, ckpt_dir, modality="all"):

    if modality == "all":
        audio_target_path = os.path.join(target_path, "audio")
        video_target_path = os.path.join(target_path, "video")
    else:
        audio_target_path = target_path
        video_target_path = target_path
    
    inference_save_path = os.path.join(args.inference_save_path, eval_id)

    audio_inference_path = os.path.join(inference_save_path, "audio")
    video_inference_path = os.path.join(inference_save_path, "video")

    # prompt 분배
    print("target_csv_files", target_csv_files)
    assert os.path.exists(target_csv_files), f"Prompt file not found: {target_csv_files}"
    all_prompts = load_prompts(target_csv_files)
    all_prompts = all_prompts[:]

    print("all_prompts length", len(all_prompts))
    num_processes = accelerator.num_processes
    prompt_subsets = split_prompts_evenly(all_prompts, num_processes)
    if accelerator.process_index < len(prompt_subsets):
        process_prompts = prompt_subsets[accelerator.process_index]
    else:
        process_prompts = []
    prompt_sublist = process_prompts

    # MMG inference
    run_inference(args, accelerator, prompt_sublist, inference_save_path, ckpt_dir)


    accelerator.wait_for_everyone()

    # MMG_EVAL
    with torch.no_grad():
        accelerator.wait_for_everyone()
        fad, clap_avg, fvd, clip_avg, imagebind_score, av_align= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


        # if modality == "all" or modality == "audio":
        #     if accelerator.is_main_process:
        #         fad, clap_avg, _ = evaluate_audio_metrics(
        #             preds_folder=audio_inference_path,
        #             target_folder=audio_target_path,
        #             metrics=['FAD','CLAP'],
        #             clap_model=1,
        #             device=accelerator.device
        #         )
        # accelerator.wait_for_everyone()

        # if modality == "all" or modality == "video":
        #     if accelerator.is_main_process:
        #         fvd, clip_avg = evaluate_video_metrics(
        #             preds_folder=video_inference_path,
        #             target_folder=video_target_path,
        #             metrics=['fvd','clip'],
        #             device=accelerator.device,
        #             num_frames=args.frames
        #         )
        # accelerator.wait_for_everyone()

        # if modality == "all":
        #     if accelerator.is_main_process:
        #         imagebind_score = evaluate_imagebind_score(
        #             inference_save_path=inference_save_path,
        #             device=accelerator.device
        #         )
        # accelerator.wait_for_everyone()

        # if modality == "all":
        #     if accelerator.is_main_process:
        #         av_align = evaluate_av_align_score(
        #             audio_inference_path=audio_inference_path,
        #             video_inference_path=video_inference_path
        #         )
        # accelerator.wait_for_everyone()



        # CAM_SCORE
        # if accelerator.is_main_process:
        #     cam_score = evaluate_video_metrics(
        #         preds_folder=video_inference_path,
        #         target_folder=video_target_path,
        #         metrics=['fvd','clip'],
        #         device=accelerator.device,
        #         num_frames=frames
        #     )

        accelerator.wait_for_everyone()

        return fad, clap_avg, fvd, clip_avg, imagebind_score, av_align#, cam_score


class CrossModalCoupledUNet(nn.Module):
    """
    A coupled UNet model that fuses features from audio and video UNets via cross-modal transformers.
    Audio and video UNets are frozen, and only cross-modal layers are trainable.
    """
    def __init__(self, audio_unet, video_unet, cross_modal_config):
        super(CrossModalCoupledUNet, self).__init__()


        # Freeze all parameters in audio_unet
        self.audio_unet = audio_unet
        for name, param in self.audio_unet.named_parameters():
            param.requires_grad = "lora" in name  # Enable only LoRA layers

        # Freeze all parameters in video_unet
        self.video_unet = video_unet
        for name, param in self.video_unet.named_parameters():
            param.requires_grad = "lora" in name  # Enable only LoRA layers

        # Count trainable parameters for each UNet
        audio_trainable_params = sum(p.numel() for p in self.audio_unet.parameters() if p.requires_grad)
        video_trainable_params = sum(p.numel() for p in self.video_unet.parameters() if p.requires_grad)

        print(f"Trainable parameters in audio_unet (LoRA): {audio_trainable_params}")
        print(f"Trainable parameters in video_unet (LoRA): {video_trainable_params}")


        self.audio_cmt = nn.ModuleList()
        self.video_cmt = nn.ModuleList()
        layer_channels = cross_modal_config['layer_channels']


        for channel in layer_channels:
            d_head = cross_modal_config.get('d_head', 64)
            n_heads = channel // d_head

            audio_transformer = CrossModalTransformer(
                in_channels=channel, n_heads=n_heads, d_head=d_head, depth=1, context_dim=channel,
                use_linear=True, use_checkpoint=True, disable_self_attn=False, img_cross_attention=False
            )
            self.initialize_cross_modal_transformer(audio_transformer)
            self.audio_cmt.append(audio_transformer)

            video_transformer = CrossModalTransformer(
                in_channels=channel, n_heads=n_heads, d_head=d_head, depth=1, context_dim=channel,
                use_linear=True, use_checkpoint=True, disable_self_attn=False, img_cross_attention=False
            )
            self.initialize_cross_modal_transformer(video_transformer)
            self.video_cmt.append(video_transformer)
        
        # cmt 모듈 zero init
        for cmt_module in self.audio_cmt:
            for param in cmt_module.parameters():
                nn.init.zeros_(param)
        for cmt_module in self.video_cmt:
            for param in cmt_module.parameters():
                nn.init.zeros_(param)


    def initialize_basic_transformer_block(self, block):
        for name, param in block.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                nn.init.zeros_(param)

    def initialize_cross_modal_transformer(self, transformer):
        if isinstance(transformer.proj_in, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(transformer.proj_in.weight)
            if transformer.proj_in.bias is not None:
                nn.init.zeros_(transformer.proj_in.bias)

        for block in transformer.transformer_blocks:
            self.initialize_basic_transformer_block(block)

        if isinstance(transformer.proj_out, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(transformer.proj_out.weight)
            if transformer.proj_out.bias is not None:
                nn.init.zeros_(transformer.proj_out.bias)

    def audio_down_blocks(self, down_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, output_states):
        # Process one down_block for the audio UNet
        if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
            for resnet, attn in zip(down_block.resnets, down_block.attentions):
                hidden_states = resnet(hidden_states, emb)
                hidden_states = attn(
                    hidden_states, encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )[0]
                output_states += (hidden_states,)
        else:
            for resnet in down_block.resnets:
                hidden_states = resnet(hidden_states, emb)
                output_states += (hidden_states,)
        return hidden_states, output_states

    def audio_mid_blocks(self, audio_unet, hidden_states, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs):
        # Process mid blocks for the audio UNet
        if hasattr(audio_unet.mid_block, "has_cross_attention") and audio_unet.mid_block.has_cross_attention:
            hidden_states = audio_unet.mid_block.resnets[0](hidden_states, emb)
            for resnet, attn in zip(audio_unet.mid_block.resnets[1:], audio_unet.mid_block.attentions):
                hidden_states = attn(
                    hidden_states, encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                hidden_states = resnet(hidden_states, emb)
        else:
            for resnet in audio_unet.mid_block.resnets:
                hidden_states = resnet(hidden_states, emb)
        return hidden_states

    def audio_up_blocks(self, up_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, res_hidden_states_tuple):
        # Process one up_block for the audio UNet
        if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
            for resnet, attn in zip(up_block.resnets, up_block.attentions):
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
                hidden_states = resnet(hidden_states, emb)
                hidden_states = attn(
                    hidden_states, encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
        else:
            for resnet in up_block.resnets:
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
                hidden_states = resnet(hidden_states, emb)
        return hidden_states

    def process_video_sublayer(self, sublayer, h, video_emb, video_context, batch_size):
        # Process a single layer of the video UNet block (down or up)
        if isinstance(sublayer, TimestepBlock):
            h = sublayer(h, video_emb, batch_size=batch_size)
        elif isinstance(sublayer, SpatialTransformer):
            h = sublayer(h, video_context)
        elif isinstance(sublayer, TemporalTransformer):
            h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)
            h = sublayer(h, video_context)
            h = rearrange(h, 'b c f h w -> (b f) c h w')
        else:
            h = sublayer(h)
        return h

    def video_down_block(self, block_idx, video_unet, h, video_emb, video_context, batch_size, hs):
        # Process a video down_block
        for sublayer in video_unet.input_blocks[block_idx]:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        if block_idx == 0 and video_unet.addition_attention:
            h = video_unet.init_attn(h, video_emb, context=video_context, batch_size=batch_size)
        hs.append(h)
        return h, hs

    def video_up_block(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
        # Process a video up_block
        h = torch.cat([h, hs.pop()], dim=1)
        for sublayer in video_unet.output_blocks[block_idx]:
            if isinstance(sublayer, Upsample):
                break
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        return h, hs

    def video_upsample(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
        # Process the Upsample layer of a video up_block
        h = self.process_video_sublayer(video_unet.output_blocks[block_idx][-1], h, video_emb, video_context, batch_size)
        return h, hs

    def linear_cmt(self, audio_hidden_states, h, index):
        
        # Cross-modal transformer step
        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k=k)
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f')

        condition_cross_audio_latent_token = cross_audio_latent_token
        condition_cross_video_latent_token = cross_video_latent_token

        # Cross-modal attention
        cross_video_latent_token = self.video_cmt[index](cross_video_latent_token, condition_cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[index](cross_audio_latent_token, condition_cross_video_latent_token)

        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k)
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a)
        
        return audio_hidden_states, h

    def forward(self, audio_latents, audio_timestep, audio_encoder_hidden_states,
                video_latents, video_timestep, video_context=None, video_fps=8,
                audio_attention_mask=None, audio_cross_attention_kwargs=None):
        # ---- Prepare Audio Branch ----
        audio_timesteps = audio_timestep
        if not torch.is_tensor(audio_timesteps):
            dtype = torch.int64 if isinstance(audio_timestep, int) else torch.float32
            audio_timesteps = torch.tensor([audio_timestep], dtype=dtype, device=audio_latents.device)
        elif audio_timesteps.dim() == 0:
            audio_timesteps = audio_timesteps[None].to(audio_latents.device)
        audio_timesteps = audio_timesteps.expand(audio_latents.shape[0])

        audio_t_emb = self.audio_unet.time_proj(audio_timesteps).to(dtype=audio_latents.dtype)
        audio_emb = self.audio_unet.time_embedding(audio_t_emb)
        if self.audio_unet.time_embed_act is not None:
            audio_emb = self.audio_unet.time_embed_act(audio_emb)
        if self.audio_unet.encoder_hid_proj is not None:
            audio_encoder_hidden_states = self.audio_unet.encoder_hid_proj(audio_encoder_hidden_states)
        audio_hidden_states = self.audio_unet.conv_in(audio_latents)
        audio_down_block_res_samples = (audio_hidden_states,)

        # ---- Prepare Video Branch ----
        video_emb = self.video_unet.time_embed(timestep_embedding(video_timestep, self.video_unet.model_channels))
        if self.video_unet.fps_cond:
            video_fps_tensor = torch.full_like(video_timestep, video_fps) if isinstance(video_fps, int) else video_fps
            video_emb += self.video_unet.fps_embedding(timestep_embedding(video_fps_tensor, self.video_unet.model_channels))
        b, _, t, _, _ = video_latents.shape
        video_context = video_context.repeat_interleave(repeats=t, dim=0) if video_context is not None else None
        video_emb = video_emb.repeat_interleave(repeats=t, dim=0)
        h = rearrange(video_latents, 'b c t h w -> (b t) c h w').type(self.video_unet.dtype)
        video_emb = video_emb.to(h.dtype)
        hs = []

        # ---- Audio & Video Down Blocks ----
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            self.audio_unet.down_blocks[0],
            audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs,
            audio_down_block_res_samples
        )
        h, hs = self.video_down_block(0, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(1, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(2, self.video_unet, h, video_emb, video_context, b, hs)

        # ---- Cross-Modal Transformer #0 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 0)

        # Audio downsample & video down_block
        audio_hidden_states = self.audio_unet.down_blocks[0].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        h, hs = self.video_down_block(3, self.video_unet, h, video_emb, video_context, b, hs)

        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            self.audio_unet.down_blocks[1],
            audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs,
            audio_down_block_res_samples
        )
        h, hs = self.video_down_block(4, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(5, self.video_unet, h, video_emb, video_context, b, hs)

        # ---- Cross-Modal Transformer #1 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 1)

        # Audio downsample & video down_block
        audio_hidden_states = self.audio_unet.down_blocks[1].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        h, hs = self.video_down_block(6, self.video_unet, h, video_emb, video_context, b, hs)

        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            self.audio_unet.down_blocks[2],
            audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs,
            audio_down_block_res_samples
        )
        h, hs = self.video_down_block(7, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(8, self.video_unet, h, video_emb, video_context, b, hs)

        # ---- Cross-Modal Transformer #2 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 2)

        # Audio downsample & video down_block
        audio_hidden_states = self.audio_unet.down_blocks[2].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        h, hs = self.video_down_block(9, self.video_unet, h, video_emb, video_context, b, hs)

        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            self.audio_unet.down_blocks[3],
            audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs,
            audio_down_block_res_samples
        )
        h, hs = self.video_down_block(10, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(11, self.video_unet, h, video_emb, video_context, b, hs)

        # ---- Mid Blocks ----
        audio_hidden_states = self.audio_mid_blocks(
            self.audio_unet, audio_hidden_states, audio_emb,
            audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs
        )
        for sublayer in self.video_unet.middle_block:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, b)

        # ---- Up Blocks ----
        # Audio up_block #0
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[0].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[0].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[0], audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs, audio_res_samples
        )
        h, hs = self.video_up_block(0, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(1, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(2, self.video_unet, h, hs, video_emb, video_context, b)

        # ---- Cross-Modal Transformer #3 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 3)

        audio_hidden_states = self.audio_unet.up_blocks[0].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(2, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio up_block #1
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[1].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[1].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[1], audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs, audio_res_samples
        )
        h, hs = self.video_up_block(3, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(4, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(5, self.video_unet, h, hs, video_emb, video_context, b)

        # ---- Cross-Modal Transformer #4 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 4)

        audio_hidden_states = self.audio_unet.up_blocks[1].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(5, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio up_block #2
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[2].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[2].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[2], audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs, audio_res_samples
        )
        h, hs = self.video_up_block(6, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(7, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(8, self.video_unet, h, hs, video_emb, video_context, b)

        # ---- Cross-Modal Transformer #5 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 5)

        audio_hidden_states = self.audio_unet.up_blocks[2].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(8, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio up_block #3
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[3].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[3].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[3], audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs, audio_res_samples
        )
        h, hs = self.video_up_block(9, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(10, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(11, self.video_unet, h, hs, video_emb, video_context, b)

        # ---- Output Layers ----
        if self.audio_unet.conv_norm_out is not None:
            audio_hidden_states = self.audio_unet.conv_norm_out(audio_hidden_states)
            if self.audio_unet.conv_act is not None:
                audio_hidden_states = self.audio_unet.conv_act(audio_hidden_states)
        audio_hidden_states = self.audio_unet.conv_out(audio_hidden_states)

        for sublayer in self.video_unet.out:
            h = sublayer(h)
        h = rearrange(h, '(b f) c h w -> b c f h w', b=b)

        return audio_hidden_states, h



def main():
    args = parse_args()



    # 로그 설정 (콘솔 출력 없이 파일에만 저장)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.tensorboard_log_dir}.log")  # 로그 파일 저장 경로
        ]
    )


    # Datasets
    dataset = AudioVideoDataset(
        csv_path=args.csv_path,
        spectrogram_dir=args.spectrogram_dir,
        video_dir=args.video_dir,
        split="train",
        slice_duration=args.duration,
        sample_rate=args.sampling_rate,
        hop_size=args.hop_size,
        target_frames=args.frames,
        target_height=args.height,
        target_width=args.width,
        ta_dir = args.ta_spectrogram_dir,
        ta_csv_path = args.ta_csv_path,
        tv_dir = args.tv_video_dir,
        tv_csv_path = args.tv_csv_path,
    )

    # dataloader
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(mixed_precision=args.dtype, gradient_accumulation_steps=args.gradient_accumulation, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    dtype = torch.bfloat16

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


    # teacher Audio Models
    teacher_audio_unet = UNet2DConditionModel.from_pretrained(args.audio_model_name, subfolder="unet")
    # LoRA config
    teacher_audio_unet_lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    teacher_audio_unet.add_adapter(teacher_audio_unet_lora_config)
    teacher_audio_unet = load_accelerator_ckpt(teacher_audio_unet, args.audio_lora_ckpt_path)

    for param in teacher_audio_unet.parameters():
        param.requires_grad = False
    teacher_audio_unet.eval()




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


    seed = args.seed
    # PyTorch Generator 설정
    generator = torch.Generator(device=device).manual_seed(seed)
    random.seed(seed)

        


    # Video UNet
    video_config = OmegaConf.load(args.videocrafter_config)
    video_model = instantiate_from_config(video_config.model)
    state_dict = torch.load(args.videocrafter_ckpt_path)['state_dict']
    video_model.load_state_dict(state_dict, strict=False) # 로라 때문에 false
    video_model.to(device=device,dtype=dtype)
    video_unet = video_model.model.diffusion_model.eval()

    video_unet = load_accelerator_ckpt(video_unet, args.video_lora_ckpt_path)


    # Video _teacher UNet
    video_teacher_config = OmegaConf.load(args.videocrafter_config)
    video_teacher_model = instantiate_from_config(video_teacher_config.model)
    state_dict_teacher = torch.load(args.videocrafter_ckpt_path)['state_dict']
    video_teacher_model.load_state_dict(state_dict_teacher, strict=False) # 로라 때문에 false
    video_teacher_model.to(device=device,dtype=dtype)
    video_teacher_unet = video_teacher_model.model.diffusion_model.eval()

    teacher_video_unet = load_accelerator_ckpt(video_teacher_unet, args.video_lora_ckpt_path)
    
    for param in teacher_video_unet.parameters():
        param.requires_grad = False
    teacher_video_unet.eval()



    ## audio lora 및 video lora 불러오기 / cmt 제외 freeze 혹은 lora까지 freeze
    # cmt zero init 필요
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device
    }

    model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config)


    noise_scheduler = DDPMScheduler.from_pretrained(args.audio_model_name, subfolder="scheduler")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))
    if accelerator.is_main_process:
        print(f"Total MMG params: {total_params}")
        print(f"Trainable MMG params: {trainable_params}")

    total_params = sum(p.numel() for p in video_unet.parameters())
    trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, video_unet.parameters()))
    if accelerator.is_main_process:
        print(f"Total video_unet params: {total_params}")
        print(f"Trainable video_unet params: {trainable_params}")

    total_params = sum(p.numel() for p in audio_unet.parameters())
    trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, audio_unet.parameters()))
    if accelerator.is_main_process:
        print(f"Total audio_unet params: {total_params}")
        print(f"Trainable audio_unet params: {trainable_params}")

    total_params = sum(p.numel() for p in teacher_video_unet.parameters())
    trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, teacher_video_unet.parameters()))
    if accelerator.is_main_process:
        print(f"Total teacher_video_unet params: {total_params}")
        print(f"Trainable teacher_video_unet params: {trainable_params}")

    total_params = sum(p.numel() for p in teacher_audio_unet.parameters())
    trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, teacher_audio_unet.parameters()))
    if accelerator.is_main_process:
        print(f"Total teacher_audio_unet params: {total_params}")
        print(f"Trainable teacher_audio_unet params: {trainable_params}")



    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    model, optimizer, dataloader, teacher_audio_unet, teacher_video_unet = accelerator.prepare(model, optimizer, dataloader, teacher_audio_unet, teacher_video_unet)

    # audio_vae, audio_image_processor, audio_text_encoder_list, audio_adapter_list, 



    # ====== 체크포인트에서 resume하는 부분 추가 ======
    start_epoch = 0
    global_step = 0

    resume_batch_idx = 0
    if args.cross_modal_checkpoint_path is not None:
        # accelerator가 저장했던 전체 상태를 로드합니다.
        accelerator.load_state(args.cross_modal_checkpoint_path)
        training_state_path = os.path.join(args.cross_modal_checkpoint_path, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            global_step = training_state.get("global_step", 0)
            # 마지막 저장된 에폭 이후부터 재개하도록 (+1)
            start_epoch = training_state.get("epoch", 0) + 1
            resume_batch_idx = global_step
            print(f"체크포인트로부터 학습 재개: 에폭 {start_epoch}부터, 글로벌 스텝 {global_step}")
        else:
            print("체크포인트 내 training_state.json 파일을 찾을 수 없어, 새롭게 시작합니다.")
    # =================================================



    model.train()
    losses = []
    losses_video = []
    losses_audio = []
    losses_distill_audio = []
    losses_distill_video = []

    full_batch_size = args.num_gpu * args.gradient_accumulation * args.train_batch_size

    # TensorBoard SummaryWriter (main process 에서만 실행)
    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    if accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="MMG_Distillation_auffusion_videocrafter", name=f"{args.date}_lr_{args.learning_rate}_batch_{full_batch_size}")
    else:
        os.environ["WANDB_MODE"] = "offline"

    currunt_idx = 0

    # gpu4 -> global_step * 2
    # global_step *= 2

    for epoch in range(args.num_epochs):
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch") as tepoch:
            
            for batch_idx, batch in enumerate(tepoch):
                currunt_idx += 1

                if currunt_idx < global_step: # and epoch+1 <= start_epoch: 
                    continue
                
                with accelerator.accumulate(model):

                    optimizer.zero_grad()

                    spec_tensors = batch["spec"].to(device=device,dtype=dtype)
                    video_tensors = batch["video_tensor"].to(device=device,dtype=dtype)
                    audio_caption = batch["tav_audio_caption"]
                    video_caption = batch["tav_video_caption"]

                    ta_spec_tensors = batch["ta_spec"].to(device=device,dtype=dtype)
                    ta_caption_list = batch["ta_caption"]
                    tv_video_tensors = batch["tv_video_tensor"].to(device=device,dtype=dtype)
                    tv_caption_list = batch["tv_caption"]

                    
                    batch_size = spec_tensors.size(0)
                    mask = (torch.rand(batch_size, 1, 1, device=device) < 0.1) # [batch_size, 1, 1] for CFG

                    # video encoding
                    video_tensors = video_tensors.permute(0, 4, 1, 2, 3)
                    with accelerator.autocast():
                        video_latents = video_model.encode_first_stage(video_tensors)
                        video_text_embed = video_model.get_learned_conditioning(video_caption)
                        video_null_prompts = batch_size * [""]
                        video_null_text_embed = video_model.get_learned_conditioning(video_null_prompts)
                        video_text_embed = torch.where(mask, video_null_text_embed, video_text_embed)
                        fps_tensor = torch.full((batch_size,), args.fps, device=device)

                    # audio encoding
                    with accelerator.autocast():
                        audio_text_embed = encode_audio_prompt(
                            text_encoder_list=audio_text_encoder_list,
                            tokenizer_list=audio_tokenizer_list,
                            adapter_list=audio_adapter_list,
                            tokenizer_model_max_length=77,
                            dtype=spec_tensors.dtype,
                            prompt=audio_caption,
                            device=accelerator.device
                        )
                        audio_null_text_emb = torch.zeros_like(audio_text_embed)  # [batch_size, 77, 768]
                        audio_text_embed = torch.where(mask, audio_null_text_emb, audio_text_embed)  # [batch_size, 77, 768]

                        image = audio_image_processor.preprocess(spec_tensors)
                        vae_output = audio_vae.encode(image)
                        audio_latents = retrieve_latents(vae_output, generator=generator)
                        audio_latents = audio_vae.config.scaling_factor * audio_latents


                    # tv_video encoding
                    tv_video_tensors = tv_video_tensors.permute(0, 4, 1, 2, 3)
                    with accelerator.autocast():
                        tv_video_latents = video_model.encode_first_stage(tv_video_tensors)
                        tv_video_text_embed = video_model.get_learned_conditioning(tv_caption_list)
                        tv_video_null_prompts = batch_size * [""]
                        tv_video_null_text_embed = video_model.get_learned_conditioning(tv_video_null_prompts)
                        tv_video_text_embed = torch.where(mask, tv_video_null_text_embed, tv_video_text_embed)
                        tv_fps_tensor = torch.full((batch_size,), args.fps, device=device)

                    # ta_audio encoding
                    with accelerator.autocast():
                        ta_audio_text_embed = encode_audio_prompt(
                            text_encoder_list=audio_text_encoder_list,
                            tokenizer_list=audio_tokenizer_list,
                            adapter_list=audio_adapter_list,
                            tokenizer_model_max_length=77,
                            dtype=ta_spec_tensors.dtype,
                            prompt=ta_caption_list,
                            device=accelerator.device
                        )
                        ta_audio_null_text_emb = torch.zeros_like(ta_audio_text_embed)  # [batch_size, 77, 768]
                        ta_audio_text_embed = torch.where(mask, ta_audio_null_text_emb, ta_audio_text_embed)  # [batch_size, 77, 768]

                        ta_image = audio_image_processor.preprocess(ta_spec_tensors)
                        ta_vae_output = audio_vae.encode(ta_image)
                        ta_audio_latents = retrieve_latents(ta_vae_output, generator=generator)
                        ta_audio_latents = audio_vae.config.scaling_factor * ta_audio_latents


                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()
                    noise_audio = torch.randn_like(audio_latents).to(device)
                    noise_video = torch.randn_like(video_latents).to(device)
                    noised_audio_latents = noise_scheduler.add_noise(audio_latents, noise_audio, timesteps)
                    noised_video_latents = video_model.q_sample(x_start=video_latents, t=timesteps, noise=noise_video)

                    ta_noise_audio = torch.randn_like(ta_audio_latents).to(device)
                    tv_noise_video = torch.randn_like(tv_video_latents).to(device)
                    ta_noised_audio_latents = noise_scheduler.add_noise(ta_audio_latents, ta_noise_audio, timesteps)
                    tv_noised_video_latents = video_model.q_sample(x_start=tv_video_latents, t=timesteps, noise=tv_noise_video)

                    # TAV Forward pass
                    audio_output, video_output = model(
                        audio_latents=noised_audio_latents,
                        audio_timestep=timesteps,
                        audio_encoder_hidden_states=audio_text_embed,
                        video_latents=noised_video_latents,
                        video_timestep=timesteps,
                        video_context=video_text_embed,
                        video_fps=fps_tensor
                    )

                    # ta_audio_output, tv_video_output = model(
                    #     audio_latents=ta_noised_audio_latents,
                    #     audio_timestep=timesteps,
                    #     audio_encoder_hidden_states=ta_audio_text_embed,
                    #     video_latents=tv_noised_video_latents,
                    #     video_timestep=timesteps,
                    #     video_context=tv_video_text_embed,
                    #     video_fps=tv_fps_tensor
                    # )

                    teacher_audio_output = teacher_audio_unet(
                        noised_audio_latents,
                        timesteps,
                        encoder_hidden_states=audio_text_embed,
                        return_dict=False,
                    )[0]

                    teacher_video_output = teacher_video_unet(
                        noised_video_latents,
                        timesteps,
                        context=video_text_embed,
                        fps=fps_tensor,
                    )

                    # Weighted loss
                    
                    loss_audio = args.audio_loss_weight * F.mse_loss(audio_output, noise_audio)
                    loss_video = args.video_loss_weight * F.mse_loss(video_output, noise_video)
                    ta_distill_loss_audio = args.ta_audio_loss_weight * F.mse_loss(audio_output, teacher_audio_output)
                    tv_distill_loss_video = args.tv_video_loss_weight * F.mse_loss(video_output, teacher_video_output)

                    loss = loss_audio + loss_video + ta_distill_loss_audio + tv_distill_loss_video
                    loss.requires_grad_(True)

                    losses.append(loss.item())
                    losses_audio.append(loss_audio.item())
                    losses_video.append(loss_video.item())
                    losses_distill_audio.append(ta_distill_loss_audio.item())
                    losses_distill_video.append(tv_distill_loss_video.item())

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    global_step += 1


                    if accelerator.is_main_process:
                        avg_losses = sum(losses) / len(losses)
                        avg_losses_video = sum(losses_video) / len(losses_video)
                        avg_losses_audio = sum(losses_audio) / len(losses_audio)
                        avg_losses_distill_audio = sum(losses_distill_audio) / len(losses_distill_audio)
                        avg_losses_distill_video = sum(losses_distill_video) / len(losses_distill_video)
                    
                        
                        # 텐서보드에 로그
                        if writer is not None:
                            writer.add_scalar("train/loss", avg_losses, global_step)
                            writer.add_scalar("train/loss_audio", avg_losses_audio, global_step)
                            writer.add_scalar("train/loss_video", avg_losses_video, global_step)
                            writer.add_scalar("train/loss_ta_distill_audio", avg_losses_distill_audio, global_step)
                            writer.add_scalar("train/loss_tv_distill_video", avg_losses_distill_video, global_step)
                            writer.add_scalar("epoch", epoch, global_step)

                        # 로그 기록
                        logging.info("######################################################################")
                        logging.info(f"train/loss: {avg_losses}, global_step: {global_step}")
                        logging.info(f"train/loss_audio: {avg_losses_audio}, global_step: {global_step}")
                        logging.info(f"train/loss_video: {avg_losses_video}, global_step: {global_step}")
                        logging.info(f"train/loss_ta_distill_audio: {avg_losses_distill_audio}, global_step: {global_step}")
                        logging.info(f"train/loss_tv_distill_video: {avg_losses_distill_video}, global_step: {global_step}")


                        wandb.log({
                            "train/loss": avg_losses,
                            "train/loss_audio": avg_losses_audio,
                            "train/loss_video": avg_losses_video,
                            "train/loss_ta_distill_audio": avg_losses_distill_audio,
                            "train/loss_tv_distill_video": avg_losses_distill_video,                            
                            "epoch": epoch,
                            "step": global_step
                        })
                        
                        losses = []
                        losses_video = []
                        losses_audio = []
                        losses_distill_audio = []
                        losses_distill_video = []


                # Save & Evaluate Checkpoints (step)
                if (global_step+1) % (args.eval_every * args.gradient_accumulation) == 0:
                    ckpt_dir = os.path.join(args.ckpt_save_path, f"checkpint_{args.tensorboard_log_dir}/checkpoint-step-{global_step}")
                    # save checkpoint
                    if accelerator.is_main_process:
                        accelerator.save_state(ckpt_dir)

                        training_state = {"global_step": global_step, "epoch": epoch, "step": global_step}
                        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                            json.dump(training_state, f)
                        print(f"[Step {global_step}] Checkpoint saved at: {ckpt_dir}")
                    accelerator.wait_for_everyone()

                    safetensor_path = os.path.join(ckpt_dir, "model.safetensors")


                    # evaluate model
                    gpt_prompt_fad, gpt_prompt_clap_avg, gpt_prompt_fvd, gpt_prompt_clip_avg, gpt_prompt_imagebind_score, gpt_prompt_av_align = evaluate_model(
                        args=args,
                        accelerator=accelerator,
                        target_csv_files="/home/work/kby_hgh/audio_video_100_prompts.csv",
                        target_path=args.vgg_gt_test_path,
                        eval_id=f"{args.infer_name}_step_{global_step}_gpt_prompt",
                        ckpt_dir=safetensor_path
                    )
                    accelerator.wait_for_everyone()

                    # evaluate model
                    vgg_fad, vgg_clap_avg, vgg_fvd, vgg_clip_avg, vgg_imagebind_score, vgg_av_align = evaluate_model(
                        args=args,
                        accelerator=accelerator,
                        target_csv_files=args.vgg_csv_path,
                        target_path=args.vgg_gt_test_path,
                        eval_id=f"{args.infer_name}_step_{global_step}_vggsound_sparse",
                        ckpt_dir=safetensor_path
                    )
                    accelerator.wait_for_everyone()

                # if (global_step+1) % (args.eval_every * args.gradient_accumulation) == 0:
                #     if (global_step+1) % (args.eval_every * args.gradient_accumulation * 2) != 0:
                #         # evaluate model
                #         vbench_fad, vbench_clap_avg, vbench_fvd, vbench_clip_avg, vbench_imagebind_score, vbench_av_align = evaluate_model(
                #             args=args,
                #             accelerator=accelerator,
                #             target_csv_files=args.vbench_csv_path,
                #             target_path=args.vbench_gt_test_path,
                #             eval_id=f"{args.infer_name}_step_{global_step}_vbench",
                #             ckpt_dir=safetensor_path
                #         )
                #         accelerator.wait_for_everyone()

                #         # evaluate model
                #         ac_fad, ac_clap_avg, ac_fvd, ac_clip_avg, ac_imagebind_score, ac_av_align = evaluate_model(
                #             args=args,
                #             accelerator=accelerator,
                #             target_csv_files=args.ac_csv_path,
                #             target_path=args.ac_gt_test_path,
                #             eval_id=f"{args.infer_name}_step_{global_step}_ac",
                #             ckpt_dir=safetensor_path
                #         )
                #         accelerator.wait_for_everyone()



    if accelerator.is_main_process and writer is not None:
        writer.close()

    print("Training Done.")


import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

