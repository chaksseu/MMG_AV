import argparse
import contextlib
import datetime
import json
import os
import re
import sys
import time
import traceback
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Optional, List, Union, Tuple
from scipy.io.wavfile import write
from einops import rearrange
from tqdm import tqdm
import wandb

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from accelerate import Accelerator
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, PNDMScheduler, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
from peft import LoraConfig

from scripts.evaluation.funcs import load_model_checkpoint#, load_prompts
from utils.utils import instantiate_from_config
from lvdm.models.utils_diffusion import (
    make_ddim_sampling_parameters, make_ddim_timesteps, timestep_embedding
)
from lvdm.common import noise_like
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer, CrossModalTransformer,
    Downsample, Upsample, TimestepBlock
)
from mmg_inference.auffusion_pipe_functions import (
    prepare_extra_step_kwargs, ConditionAdapter,
    import_model_class_from_model_name_or_path, Generator
)


# from mmg_training.train_MMG_Model_0223_MMG_LoRA import CrossModalCoupledUNet




def load_prompts(prompt_file: str):
    """
    CSV 파일에서 'split'이 'test'인 행의 'caption'을 불러오는 함수
    """
    prompts = []
    try:
        with open(prompt_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                caption = row.get('caption', '').strip()
                if caption:
                    prompts.append(caption)
    except Exception as e:
        print(f"Error reading prompt file: {e}")

    
    return prompts



#### CMT ####
class CrossModalCoupledUNet(nn.Module):
    """
    A coupled UNet model that fuses features from audio and video UNets via cross-modal transformers.
    Audio and video UNets are frozen, and only cross-modal layers are trainable.
    """
    def __init__(self, audio_unet, video_unet, cross_modal_config, device, dtype):
        super(CrossModalCoupledUNet, self).__init__()
        # Freeze audio_unet

        self.dtype = dtype
        self.device = device

        self.audio_unet = audio_unet.to(device=device, dtype=dtype)
        # self.audio_unet.dtype=dtype
        for name, param in self.audio_unet.named_parameters():
            param.requires_grad = False

        # Freeze video_unet
        self.video_unet = video_unet.to(device=device, dtype=dtype)
        for name, param in self.video_unet.named_parameters():
            param.requires_grad = False
        self.video_unet.dtype=dtype

        # for name, param in self.video_unet.named_parameters():
        #     if 'lora_block' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False


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

        video_emb = self.video_unet.time_embed(timestep_embedding(video_timestep, self.video_unet.model_channels).to(self.dtype))

        if self.video_unet.fps_cond:
            video_fps_tensor = torch.full_like(video_timestep, video_fps) if isinstance(video_fps, int) else video_fps
            video_emb += self.video_unet.fps_embedding(timestep_embedding(video_fps_tensor, self.video_unet.model_channels).to(self.dtype))
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

#############


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
    
    savepath = os.path.join(savedir, f"{base_filename}.mp4")
    
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
    audio_unet,
    video_unet,
    checkpoint_path,
    device,
    dtype,
) -> CrossModalCoupledUNet:
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device
    }
    cross_modal_model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config, device, dtype)
    checkpoint = load_file(checkpoint_path)
    cross_modal_model.load_state_dict(checkpoint)
    cross_modal_model = cross_modal_model.to(device, dtype).eval()
    return cross_modal_model


@torch.no_grad()
def run_inference(
    args,
    accelerator,
    prompt_sublist,
    inference_save_path,
    ckpt_dir
):
    try:
        print("Start Inference")

        # Set unique seed per process
        unique_seed = args.seed + accelerator.process_index
        seed_everything(unique_seed)
        device = accelerator.device

        dtype =  torch.float32
        generator = torch.Generator(device=device).manual_seed(unique_seed)


        if not prompt_sublist:
            accelerator.print(f"Process {accelerator.process_index}: No prompts to process.")
            return

        audio_length = int(args.duration * 16000)
        latent_time = int(12.5 * args.duration)

        do_audio_cfg = args.audio_guidance_scale > 1.0
        do_video_cfg = args.video_unconditional_guidance_scale > 1.0

        # Load video pipeline
        config = OmegaConf.load(args.videocrafter_config)
        model_config = config.pop("model", OmegaConf.create())
        video_pipeline = instantiate_from_config(model_config).to(device)
        assert os.path.exists(args.videocrafter_ckpt_path), f"Checkpoint [{args.videocrafter_ckpt_path}] not found!"
        video_pipeline = load_model_checkpoint(video_pipeline, args.videocrafter_ckpt_path, full_strict=False)
        video_pipeline.eval()
        video_unet = video_pipeline.model.diffusion_model.to(device, dtype)


        # Model directory
        model_dir = args.audio_model_name
        if not os.path.isdir(model_dir):
            model_dir = snapshot_download(model_dir)

        # Load audio components
        vocoder = load_vocoder(model_dir, device, dtype)
        vae, image_processor = load_vae(model_dir, device, dtype)
        text_encoders, tokenizers, adapters = load_text_encoders(model_dir, device, dtype)
        
        audio_unet = load_audio_unet(model_dir, device, dtype)
            # LoRA config
        lora_config = LoraConfig(
            r=128,
            lora_alpha=128,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        audio_unet.add_adapter(lora_config)



        # Load MMG components
        cross_modal_model = load_cross_modal_unet(audio_unet, video_unet, ckpt_dir, device, dtype)

 


        assert (args.height % 16 == 0) and (args.width % 16 == 0), "Video dimensions must be multiples of 16!"
        latent_h, latent_w = args.height // 8, args.width // 8
        frames = video_pipeline.temporal_length if args.frames < 0 else args.frames
        channels = video_pipeline.channels

        # Create output dirs (main process only)
        if accelerator.is_main_process:
            os.makedirs(inference_save_path, exist_ok=True)
            audio_dir = os.path.join(inference_save_path, "audio")
            video_dir = os.path.join(inference_save_path, "video")
            combined_dir = os.path.join(inference_save_path, "combined_video")
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(combined_dir, exist_ok=True)

        accelerator.wait_for_everyone()

        total_prompts = len(prompt_sublist)
        num_batches = (total_prompts + args.inference_batch_size - 1) // args.inference_batch_size

        # Initialize DDIM sampler for video
        sampler_state = initialize_ddim_sampler(video_pipeline)
        make_ddim_schedule(sampler_state, ddim_num_steps=args.num_inference_steps, ddim_eta=args.video_ddim_eta, verbose=False)

        # Setup progress bar on main process
        if accelerator.is_main_process:
            pbar = tqdm(total=num_batches, desc="Generating", disable=not accelerator.is_main_process)
        else:
            pbar = None



        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.inference_batch_size
            end_idx = min(start_idx + args.inference_batch_size, total_prompts)
            current_batch_size = end_idx - start_idx
            current_prompts = prompt_sublist[start_idx:end_idx]


            # audio_prompts = []
            # video_prompts = []

            # for i in range(len(current_prompts)):
            #     audio_prompts.append("a sound of " + current_prompts[i])
            #     video_prompts.append("a video of " + current_prompts[i])


            video_noise_shape = [current_batch_size, channels, frames, latent_h, latent_w]
            fps_tensor = torch.tensor([args.fps] * current_batch_size).to(device).long()

            # Get text embedding for video
            video_text_emb = video_pipeline.get_learned_conditioning(current_prompts) ###
            video_cond = {"c_crossattn": [video_text_emb], "fps": fps_tensor}

            # Unconditional video
            cfg_scale = args.video_unconditional_guidance_scale
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
                dtype=video_latents.dtype,
                prompt=current_prompts,
                device=device,
                do_classifier_free_guidance=do_audio_cfg
            )

            # Initialize audio latents
            audio_latent_shape = (current_batch_size, 4, 32, latent_time)
            audio_latents = randn_tensor(audio_latent_shape, generator=generator, device=device, dtype=video_latents.dtype)
            audio_latents *= audio_scheduler.init_noise_sigma

            extra_step_kwargs = prepare_extra_step_kwargs(audio_scheduler, generator, args.audio_ddim_eta)
            timesteps = sampler_state['ddim_timesteps']
            total_steps = timesteps.shape[0]
            time_range = np.flip(timesteps)


            # Denoising loop
            for step_idx, (video_step, audio_step) in enumerate(zip(time_range, audio_timesteps)):
                index = total_steps - step_idx - 1
                video_ts = torch.full((current_batch_size,), video_step, device=device, dtype=video_latents.dtype)


                # CFG for audio/video
                if do_audio_cfg or do_video_cfg:
                    neg_audio_prompt_embeds = torch.zeros_like(audio_prompt_embeds, dtype=video_latents.dtype, device=device)
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
                    audio_out = audio_out_uncond + args.audio_guidance_scale * (audio_out - audio_out_uncond)
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
            audio_dir = os.path.join(inference_save_path, "audio")
            video_dir = os.path.join(inference_save_path, "video")
            combined_dir = os.path.join(inference_save_path, "combined_video")

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

                # # 오디오가 포함된 비디오 파일 저장
                # save_video_with_audio(
                #     video_path=video_filepath,
                #     audio_path=audio_filepath,
                #     savedir=combined_dir,
                #     base_filename=base_filename,
                #     fps=args.fps
                # )

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        accelerator.print(f"Process {accelerator.process_index}: Completed inference. Results saved in {inference_save_path}.")

    except Exception as e:
        accelerator.print(f"Process {accelerator.process_index}: Encountered an error.")
        traceback.print_exc()


import argparse
import os
import torch
from accelerate import Accelerator

def get_parser():
    """ 
    스크립트 실행 시 필요한 인자들을 정의하고 반환합니다.
    필요에 따라 default 값이나 help 문자열을 자유롭게 조정하세요.
    """
    parser = argparse.ArgumentParser(description="Multi-modal Generation Inference")

    # 기본 설정
    parser.add_argument("--prompt_file", type=str, default="",
                        help="프롬프트가 저장된 텍스트 파일 경로")
    parser.add_argument("--inference_save_path", type=str, default="/home/work/kby_hgh/",
                        help="결과물을 저장할 경로(디렉토리)")
    parser.add_argument("--ckpt_dir", type=str, default="/home/work/kby_hgh/MMG_CHECKPOINT/checkpint_tensorboard/",
                        help="CrossModalCoupledUNet의 safetensors 체크포인트 파일 경로")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드를 설정합니다.")
    
    # 비디오 Crafter 관련 설정
    parser.add_argument("--videocrafter_config", type=str, default="configs/inference_t2v_512_v2.0.yaml",
                        help="VideoCrafter 모델의 config 파일(.yaml) 경로")
    parser.add_argument("--videocrafter_ckpt_path", type=str, default="scripts/evaluation/model.ckpt",
                        help="VideoCrafter 모델의 checkpoint(.ckpt) 경로")
    
    # Audio 모델 관련 설정
    parser.add_argument("--audio_model_name", type=str, default="auffusion/auffusion-full",
                        help="Hugging Face나 로컬 디렉토리 상의 audio diffusion 모델 경로/이름")

    # 생성 파라미터
    parser.add_argument("--duration", type=float, default=3.2,
                        help="생성할 오디오 길이(초 단위)")
    parser.add_argument("--fps", type=float, default=12.5,
                        help="생성할 영상의 FPS(frames per second)")
    parser.add_argument("--frames", type=int, default=40,
                        help="영상의 프레임 수(기본 -1이면 VideoCrafter 기본 길이 사용)")
    parser.add_argument("--height", type=int, default=320,
                        help="생성할 영상의 세로 해상도(16의 배수 권장)")
    parser.add_argument("--width", type=int, default=512,
                        help="생성할 영상의 가로 해상도(16의 배수 권장)")

    # 오디오/비디오 CFG 스케일
    parser.add_argument("--audio_guidance_scale", type=float, default=7.5,
                        help="오디오 CFG 스케일 (1.0 초과일 때만 CFG 적용)")
    parser.add_argument("--video_unconditional_guidance_scale", type=float, default=12.0,
                        help="비디오 uncond CFG 스케일 (1.0 초과일 때만 CFG 적용)")

    # DDIM(혹은 다른) 스케줄러 설정
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Inference 시 확산 스텝 수")
    parser.add_argument("--audio_ddim_eta", type=float, default=0.0,
                        help="오디오 확산 시 DDIM eta 값")
    parser.add_argument("--video_ddim_eta", type=float, default=0.0,
                        help="비디오 확산 시 DDIM eta 값")
    
    # 배치 처리
    parser.add_argument("--inference_batch_size", type=int, default=2,
                        help="프롬프트별 배치 크기")

    return parser


def main():
    # 인자 파서 준비
    parser = get_parser()
    args = parser.parse_args()

    model_name_list = [0]
    # checkpoint_dir_list = ["checkpoint-step-20287", "checkpoint-step-40575", "checkpoint-step-60863", "checkpoint-step-38399", "checkpoint-step-47999", "checkpoint-step-57599", "checkpoint-step-67199", "checkpoint-step-76799", "checkpoint-step-86399"]
    # checkpoint_dir_list = ["checkpoint-step-10143", "checkpoint-step-30431", "checkpoint-step-50719", "checkpoint-step-71007", "checkpoint-step-91295"]
    
    
    
    # dataset_list = ["OOD_gpt_prompt"]
    
    checkpoint_dir_list = ["91295"] #, "checkpoint-step-40575", "checkpoint-step-60863", "checkpoint-step-81151", "checkpoint-step-101439"]

    dataset_list = ["vbench", "ac"]
    dataset_list = ["vbench"]

    for checkpoint_dir in checkpoint_dir_list:
        for dataset in dataset_list:
            # clotho
            # panda70m
            # audiocaps
            # webvid
            # vbench
            if dataset == "vbench":
                args.prompt_file = "/home/work/kby_hgh/vbench_all_captions.csv"
            if dataset == "ac":
                args.prompt_file = "/home/work/kby_hgh/MMG_AC_test_dataset/0407_one_cap_AC_test.csv"
            if dataset == "OOD_gpt_prompt":
                args.prompt_file = "/home/work/kby_hgh/audio_video_100_prompts.csv"

            # Panda-70M # /home/work/kby_hgh/processed_csv_files/0409_onecap_processed_panda_70m_test.csv
            # AudioCaps # /home/work/kby_hgh/MMG_AC_test_dataset/0407_one_cap_AC_test.csv
            # Clotho # /home/work/kby_hgh/MMG_clotho_test_set/clotho_captions_evaluation.csv
            # VBench # /home/work/kby_hgh/vbench_all_captions.csv

            if checkpoint_dir == "checkpoint-step-10143":
                ckpt_dir = "/home/work/kby_hgh/MMG_CHECKPOINT/checkpint_tensorboard/0419_MMG_OURS_1e-4_8gpu_videocaption/checkpoint-step-10143"
            else:
                ckpt_dir = f"/home/work/kby_hgh/MMG_CHECKPOINT/checkpint_tensorboard/0420_MMG_OURS_1e-4_8gpu_videocaption_continue/checkpoint-step-{checkpoint_dir}"

            # ckpt_dir = f"/home/work/kby_hgh/MMG_CHECKPOINT/checkpint_tensorboard/0501_1138_MMG_NAIVE_DISTILL_continue_1e-4_ta_tv_weight_1_1/checkpoint-step-{checkpoint_dir}"
            
            # ckpt_dir = f"/home/work/kby_hgh/MMG_CHECKPOINT/checkpint_tensorboard/0429_1625_MMG_RC_DISTILL_1e-4_ta_tv_weight_1_1/checkpoint-step-{checkpoint_dir}"

            eval_id = "MMG_OURS_{dataset}_checkpoint-step-{checkpoint_dir}"
            eval_id = f"MMG_NAIVE_DISTILL_step_{checkpoint_dir}_{dataset}"
            # eval_id = f"MMG_RC_DISTILL_{dataset}_{checkpoint_dir}"
            
            # prompt 파일 존재 여부 확인
            assert os.path.exists(args.prompt_file), f"Prompt file not found: {args.prompt_file}"

            # Accelerate 초기화 (mixed_precision은 필요에 맞게 조정)
            # accelerator = Accelerator(mixed_precision="bf16")
            from accelerate import DistributedDataParallelKwargs
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

            accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
            # 전체 프롬프트 로드
            all_prompts = load_prompts(args.prompt_file)

            # seed_everything(args.seed)
            num_processes = accelerator.num_processes

            prompt_subsets = split_prompts_evenly(all_prompts, num_processes)
            if accelerator.process_index < len(prompt_subsets):
                prompt_sublist = prompt_subsets[accelerator.process_index]
            else:
                prompt_sublist = []

            inference_save_path = os.path.join(args.inference_save_path, eval_id)


            safetensor_path = os.path.join(ckpt_dir, "model.safetensors")


            # run_inference 실행
            run_inference(
                args=args,
                accelerator=accelerator,
                prompt_sublist=prompt_sublist,
                inference_save_path=inference_save_path,
                ckpt_dir=safetensor_path
            )

            # 모든 프로세스 동기화
            accelerator.wait_for_everyone()
            accelerator.print("Inference finished!")


if __name__ == "__main__":
    main()
