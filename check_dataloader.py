import torch
import numpy as np
from tqdm import tqdm

import torch
from diffusers import PNDMScheduler, UNet2DConditionModel, DDPMScheduler  
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer, CrossModalTransformer,
    Downsample, Upsample, TimestepBlock
)
from lvdm.common import checkpoint
from einops import rearrange
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

import torch.nn as nn
from tqdm import tqdm  


import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from diffusers import PNDMScheduler
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor


import wandb 



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


def normalize_data(data, data_min, data_max):
    # Normalize to [-1, 1]
    return 2 * (data - data_min) / (data_max - data_min) - 1

def denormalize_data(data, data_min, data_max):
    # Reverse normalization for reconstruction
    return (data + 1) / 2 * (data_max - data_min) + data_min


class LatentsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file):
        import pandas as pd
        self.root_dir = root_dir
        self.data_info = pd.read_csv(csv_file)
        
        # Define subdirectories
        self.video_latents_dir = os.path.join(root_dir, "video_latents")
        self.audio_latents_dir = os.path.join(root_dir, "audio_latents")
        self.video_text_embeds_dir = os.path.join(root_dir, "video_text_embeds")
        self.audio_text_embeds_dir = os.path.join(root_dir, "audio_text_embeds")

        # Latent statistics (replace with your actual stats)
        self.audio_stats = {
            "min": -13.403627395629883,
            "max": 10.519368171691895
        }
        self.video_stats = {
            "min": -8.698715209960938,
            "max": 8.797478675842285
        }
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Retrieve file names from CSV
        video_file = self.data_info.iloc[idx]["Video"]
        audio_file = self.data_info.iloc[idx]["Audio"]
        video_text_embed_file = self.data_info.iloc[idx]["Video_text_embed"]
        audio_text_embed_file = self.data_info.iloc[idx]["Audio_text_embed"]
        
        # Load latent tensors
        video_latent = torch.load(os.path.join(self.video_latents_dir, video_file))
        audio_latent = torch.load(os.path.join(self.audio_latents_dir, audio_file))
        video_text_embed = torch.load(os.path.join(self.video_text_embeds_dir, video_text_embed_file))
        audio_text_embed = torch.load(os.path.join(self.audio_text_embeds_dir, audio_text_embed_file))
        
        # Normalize latents to [-1, 1]
        video_latent = normalize_data(video_latent, self.video_stats["min"], self.video_stats["max"])
        audio_latent = normalize_data(audio_latent, self.audio_stats["min"], self.audio_stats["max"])
        
        sample = {
            "video_latent": video_latent,           # shape example: (B, C, T, H, W)
            "audio_latent": audio_latent,           # shape example: (B, C, H', W')
            "video_text_embed": video_text_embed,   # (B, seq_len, dim)
            "audio_text_embed": audio_text_embed    # (B, seq_len, dim)
        }
        return sample


def get_dataloader(root_dir, csv_file, batch_size=4, shuffle=True, num_workers=0):
    dataset = LatentsDataset(root_dir=root_dir, csv_file=csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


class CrossModalCoupledUNet(nn.Module):
    def __init__(self, audio_unet, video_unet, cross_modal_config):
        super(CrossModalCoupledUNet, self).__init__()
        # Initialize audio_unet and freeze its parameters
        self.audio_unet = audio_unet
        for param in self.audio_unet.parameters():
            param.requires_grad = False

        # Initialize video_unet and freeze its parameters
        self.video_unet = video_unet
        for param in self.video_unet.parameters():
            param.requires_grad = False

        # Initialize CrossModalTransformers
        self.audio_cmt = nn.ModuleList()
        self.video_cmt = nn.ModuleList()
        layer_channels = cross_modal_config['layer_channels']

        for channel in layer_channels:
            d_head = cross_modal_config.get('d_head', 64)
            n_heads = channel // d_head
            
            # Create a separate transformer for audio_cmt
            audio_transformer = CrossModalTransformer(
                in_channels=channel,
                n_heads=n_heads,
                d_head=d_head,
                depth=1,
                context_dim=channel,
                use_linear=True,
                use_checkpoint=True,
                disable_self_attn=False,
                img_cross_attention=False
            )
            self.initialize_cross_modal_transformer(audio_transformer)
            self.audio_cmt.append(audio_transformer)
            
            # Create a separate transformer for video_cmt
            video_transformer = CrossModalTransformer(
                in_channels=channel,
                n_heads=n_heads,
                d_head=d_head,
                depth=1,
                context_dim=channel,
                use_linear=True,
                use_checkpoint=True,
                disable_self_attn=False,
                img_cross_attention=False
            )
            self.initialize_cross_modal_transformer(video_transformer)
            self.video_cmt.append(video_transformer)

        
    def initialize_basic_transformer_block(self, block):
        """
        Initializes the weights of a BasicTransformerBlock.
        """
        for name, param in block.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                nn.init.zeros_(param)
                    
    def initialize_cross_modal_transformer(self, transformer):
        """
        Initializes the weights of a CrossModalTransformer.
        """
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
        if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
            # With CrossAttention
            for resnet, attn in zip(down_block.resnets, down_block.attentions):
      #          print("    [down_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
     #          print("    [down_block][attention]")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )[0]
                output_states += (hidden_states,)
        else:
            # Without CrossAttention
            for resnet in down_block.resnets:
   #             print("    [down_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
                output_states += (hidden_states,)
        return hidden_states, output_states


    def audio_mid_blocks(self, audio_unet, hidden_states, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs):
        if hasattr(audio_unet.mid_block, "has_cross_attention") and audio_unet.mid_block.has_cross_attention:
    #        print("  [mid_block][resnets]")
            hidden_states = audio_unet.mid_block.resnets[0](hidden_states, emb)
            for resnet, attn in zip(audio_unet.mid_block.resnets[1:], audio_unet.mid_block.attentions):
      #          print("  [mid_block][attention]")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
   #             print("  [mid_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
        else:
            for resnet in audio_unet.mid_block.resnets:
           #     print("  [mid_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
        return hidden_states


    def audio_up_blocks(self, up_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, res_hidden_states_tuple):
        if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
            for resnet, attn in zip(up_block.resnets, up_block.attentions):
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
       #         print("    [up_block][resnet]")
                hidden_states = resnet(hidden_states, emb)

       #         print("    [up_block][attention]")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
        else:
            for resnet in up_block.resnets:
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
    #            print("    [up_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
        return hidden_states


    def video_up_block(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
     #   print(f"  [video_output_blocks][{block_idx}]")
        h = torch.cat([h, hs.pop()], dim=1)
        for sublayer in video_unet.output_blocks[block_idx]:
            if isinstance(sublayer, Upsample):
                break
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        return h, hs


    def video_upsample(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
  #      print(f"  [video_Upsample][{block_idx}]")
        h = self.process_video_sublayer(video_unet.output_blocks[block_idx][-1], h, video_emb, video_context, batch_size)
        return h, hs


    def video_down_block(self, block_idx, video_unet, h, video_emb, video_context, batch_size, hs):
     #   print(f"  [video_input_blocks][{block_idx}]")
     #   print("Shape of h:", h.shape)
    #    print("Shape of video_emb:", video_emb.shape)
    #    print("Shape of video_context:", video_context.shape)

        for sublayer in video_unet.input_blocks[block_idx]:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        if block_idx == 0 and video_unet.addition_attention:
            h = video_unet.init_attn(h, video_emb, context=video_context, batch_size=batch_size)
        hs.append(h)
        return h, hs


    def process_video_sublayer(self, sublayer, h, video_emb, video_context, batch_size):
        layer_type = type(sublayer).__name__
  #      print(f"    [blocks]: {layer_type}")
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
    
    def forward(self, audio_latents, audio_timestep, audio_encoder_hidden_states,
                video_latents, video_timestep, video_context=None, video_fps=8,
                audio_attention_mask=None, audio_cross_attention_kwargs=None):
        ###### Prepare Audio ######
        #if self.audio_unet.config.center_input_sample:
   #         print("Centering input sample")
            #audio_latents = 2 * audio_latents - 1.0
   #     print("Processing time embeddings")
        audio_timesteps = audio_timestep
        if not torch.is_tensor(audio_timesteps):
            is_mps = audio_latents.device.type == 'mps'
            dtype = torch.float32 if is_mps else torch.float64 if isinstance(audio_timestep, float) else torch.int32 if is_mps else torch.int64
            audio_timesteps = torch.tensor([audio_timestep], dtype=dtype, device=audio_latents.device)
        elif audio_timesteps.dim() == 0:
            audio_timesteps = audio_timesteps[None].to(audio_latents.device)
        audio_timesteps = audio_timesteps.expand(audio_latents.shape[0])
        audio_t_emb = self.audio_unet.time_proj(audio_timesteps).to(dtype=audio_latents.dtype)
        audio_emb = self.audio_unet.time_embedding(audio_t_emb)
        if self.audio_unet.time_embed_act is not None:
            audio_emb = self.audio_unet.time_embed_act(audio_emb)
    #    print("Processing encoder hidden states")
        if self.audio_unet.encoder_hid_proj is not None:
            audio_encoder_hidden_states = self.audio_unet.encoder_hid_proj(audio_encoder_hidden_states)
     #   print("Processing conv_in")
        audio_hidden_states = self.audio_unet.conv_in(audio_latents)
        audio_down_block_res_samples = (audio_hidden_states,)

        ###### Prepare Video ######
        video_emb = self.video_unet.time_embed(timestep_embedding(video_timestep, self.video_unet.model_channels))
        if self.video_unet.fps_cond:
            video_fps_tensor = torch.full_like(video_timestep, video_fps) if isinstance(video_fps, int) else video_fps
            video_emb += self.video_unet.fps_embedding(timestep_embedding(video_fps_tensor, self.video_unet.model_channels))
        b, _, t, _, _ = video_latents.shape
        video_context = video_context.repeat_interleave(repeats=t, dim=0) if video_context is not None else None
        video_emb = video_emb.repeat_interleave(repeats=t, dim=0) # t -> t*b
        h = rearrange(video_latents, 'b c t h w -> (b t) c h w').type(self.video_unet.dtype)
        video_emb = video_emb.to(h.dtype)
        hs = []
        
        ####### Down Blocks ######
        #print("Processing down_blocks")
     #   print("  [audio_down_block] 0")
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            down_block=self.audio_unet.down_blocks[0],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            output_states=audio_down_block_res_samples
        )
        h, hs = self.video_down_block(0, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(1, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(2, self.video_unet, h, video_emb, video_context, b, hs)
        
        #################################################### Cross-Modal Transformer ###################################################

        #print("  [CrossModalTransformers] 0")

        #print("audio_hidden_states", audio_hidden_states.shape)
        #print("h", h.shape)

        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

        cross_video_latent_token = self.video_cmt[0](cross_video_latent_token, cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[0](cross_audio_latent_token, cross_video_latent_token)
        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
        ##################################################################################################################################

   #     print("    [audio_down_block][downsampler]")
        audio_hidden_states = self.audio_unet.down_blocks[0].downsamplers[0](audio_hidden_states)  # Audio downsample
        audio_down_block_res_samples += (audio_hidden_states,)
        h, hs = self.video_down_block(3, self.video_unet, h, video_emb, video_context, b, hs)  # Video downsample
    
   #     print("  [audio_down_block] 1")
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            down_block=self.audio_unet.down_blocks[1],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            output_states=audio_down_block_res_samples
        )
        h, hs = self.video_down_block(4, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(5, self.video_unet, h, video_emb, video_context, b, hs)

        #################################################### Cross-Modal Transformer ###################################################

      #  print("  [CrossModalTransformers] 1")

    #    print("audio_hidden_states", audio_hidden_states.shape)
#        print("h", h.shape)

        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

        cross_video_latent_token = self.video_cmt[1](cross_video_latent_token, cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[1](cross_audio_latent_token, cross_video_latent_token)
        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
        ################################################################################################################################
        
    #    print("    [audio_down_block][downsampler]")
        audio_hidden_states = self.audio_unet.down_blocks[1].downsamplers[0](audio_hidden_states)  # Audio downsample
        audio_down_block_res_samples += (audio_hidden_states,)   
        h, hs = self.video_down_block(6, self.video_unet, h, video_emb, video_context, b, hs)  # Video downsample
        
     #   print("  [audio_down_block] 2")
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            down_block=self.audio_unet.down_blocks[2],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            output_states=audio_down_block_res_samples
        )
        h, hs = self.video_down_block(7, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(8, self.video_unet, h, video_emb, video_context, b, hs)
        
        #################################################### Cross-Modal Transformer ###################################################

  #      print("  [CrossModalTransformers] 2")

   #     print("audio_hidden_states", audio_hidden_states.shape)
    #    print("h", h.shape)

        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

        cross_video_latent_token = self.video_cmt[2](cross_video_latent_token, cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[2](cross_audio_latent_token, cross_video_latent_token)
        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
        ################################################################################################################################
        
   #     print("    [audio_down_block][downsampler]")
        audio_hidden_states = self.audio_unet.down_blocks[2].downsamplers[0](audio_hidden_states)  # Audio downsample
        audio_down_block_res_samples += (audio_hidden_states,)   
        h, hs = self.video_down_block(9, self.video_unet, h, video_emb, video_context, b, hs)  # Video downsample    
        
   #     print("  [audio_down_block] 3")
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            down_block=self.audio_unet.down_blocks[3],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            output_states=audio_down_block_res_samples
        )
        h, hs = self.video_down_block(10, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(11, self.video_unet, h, video_emb, video_context, b, hs)


        ####### Mid Block ######
    #    print("Processing mid_block")
        audio_hidden_states = self.audio_mid_blocks(
            self.audio_unet, audio_hidden_states, audio_emb,
            audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs
        )
        for sublayer in self.video_unet.middle_block:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, b)


        ####### Up Blocks ######
   #     print("Processing up_blocks")
    #    print("  [audio_up_block] 0")
        if not audio_down_block_res_samples:
            raise ValueError("No residual samples available for up_block")
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[0].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[0].resnets)]
        audio_res_hidden_states_tuple = audio_res_samples 
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[0],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            res_hidden_states_tuple=audio_res_hidden_states_tuple
        )
        h, hs = self.video_up_block(0, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(1, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(2, self.video_unet, h, hs, video_emb, video_context, b)  # R,U -> R

        #################################################### Cross-Modal Transformer ###################################################

      #  print("  [CrossModalTransformers] 3")

      #  print("audio_hidden_states", audio_hidden_states.shape)
      #  print("h", h.shape)

        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

        cross_video_latent_token = self.video_cmt[3](cross_video_latent_token, cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[3](cross_audio_latent_token, cross_video_latent_token)
        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
        ################################################################################################################################

       # print("    [audio_up_block][upsampler]")
        audio_hidden_states = self.audio_unet.up_blocks[0].upsamplers[0](audio_hidden_states)  # Audio upsample
        h, hs = self.video_upsample(2, self.video_unet, h, hs, video_emb, video_context, b)  # Video upsample

        #print("  [audio_up_block] 1")
        if not audio_down_block_res_samples:
            raise ValueError("No residual samples available for up_block")
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[1].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[1].resnets)]
        audio_res_hidden_states_tuple = audio_res_samples 
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[1],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            res_hidden_states_tuple=audio_res_hidden_states_tuple
        )
        h, hs = self.video_up_block(3, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(4, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(5, self.video_unet, h, hs, video_emb, video_context, b)  # R,S,T,U -> R,S,T
        
        #################################################### Cross-Modal Transformer ###################################################

        #print("  [CrossModalTransformers] 4")

        #print("audio_hidden_states", audio_hidden_states.shape)
        #print("h", h.shape)

        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

        cross_video_latent_token = self.video_cmt[4](cross_video_latent_token, cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[4](cross_audio_latent_token, cross_video_latent_token)
        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
        ################################################################################################################################

        #print("    [audio_up_block][upsampler]")
        audio_hidden_states = self.audio_unet.up_blocks[1].upsamplers[0](audio_hidden_states)  # Audio upsample
        h, hs = self.video_upsample(5, self.video_unet, h, hs, video_emb, video_context, b)  # Video upsample

        #print("  [audio_up_block] 2")
        if not audio_down_block_res_samples:
            raise ValueError("No residual samples available for up_block")
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[2].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[2].resnets)]
        audio_res_hidden_states_tuple = audio_res_samples 
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[2],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            res_hidden_states_tuple=audio_res_hidden_states_tuple
        )
        h, hs = self.video_up_block(6, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(7, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(8, self.video_unet, h, hs, video_emb, video_context, b)  # R,S,T,U -> R,S,T
        
        #################################################### Cross-Modal Transformer ###################################################

        #print("  [CrossModalTransformers] 5")

        #print("audio_hidden_states", audio_hidden_states.shape)
        #print("h", h.shape)

        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

        cross_video_latent_token = self.video_cmt[5](cross_video_latent_token, cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[5](cross_audio_latent_token, cross_video_latent_token)
        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
        ################################################################################################################################

        #print("    [audio_up_block][upsampler]")
        audio_hidden_states = self.audio_unet.up_blocks[2].upsamplers[0](audio_hidden_states)  # Audio upsample
        h, hs = self.video_upsample(8, self.video_unet, h, hs, video_emb, video_context, b)  # Video upsample

        #print("  [audio_up_block] 3")
        if not audio_down_block_res_samples:
            raise ValueError("No residual samples available for up_block")
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[3].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[3].resnets)]
        audio_res_hidden_states_tuple = audio_res_samples 
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[3],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            res_hidden_states_tuple=audio_res_hidden_states_tuple
        )
        h, hs = self.video_up_block(9, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(10, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(11, self.video_unet, h, hs, video_emb, video_context, b)


        ###### Output Layers ######
        #print("Processing output layers")
        if self.audio_unet.conv_norm_out is not None:
            audio_hidden_states = self.audio_unet.conv_norm_out(audio_hidden_states)
            if self.audio_unet.conv_act is not None:
                audio_hidden_states = self.audio_unet.conv_act(audio_hidden_states)
        audio_hidden_states = self.audio_unet.conv_out(audio_hidden_states)
        
        for sublayer in self.video_unet.out:
            h = sublayer(h)
        h = rearrange(h, '(b f) c h w -> b c f h w', b=b)

        audio_output, video_output = audio_hidden_states, h

        return audio_output, video_output

def compute_statistics(dataloader):
    # 각 데이터셋에 대해 평균, 분산, 최대값, 최소값 계산
    audio_latent_sum = 0
    audio_latent_square_sum = 0
    audio_latent_max = float('-inf')
    audio_latent_min = float('inf')
    
    video_latent_sum = 0
    video_latent_square_sum = 0
    video_latent_max = float('-inf')
    video_latent_min = float('inf')

    num_audio_samples = 0
    num_video_samples = 0

    # 데이터셋 순회
    for batch in tqdm(dataloader, desc="Computing statistics", unit="batch"):
        audio_latents = batch["audio_latent"]
        video_latents = batch["video_latent"]

        # Audio Latent 통계 계산
        num_audio_samples += audio_latents.numel()
        audio_latent_sum += audio_latents.sum().item()
        audio_latent_square_sum += (audio_latents**2).sum().item()
        audio_latent_max = max(audio_latent_max, audio_latents.max().item())
        audio_latent_min = min(audio_latent_min, audio_latents.min().item())

        # Video Latent 통계 계산
        num_video_samples += video_latents.numel()
        video_latent_sum += video_latents.sum().item()
        video_latent_square_sum += (video_latents**2).sum().item()
        video_latent_max = max(video_latent_max, video_latents.max().item())
        video_latent_min = min(video_latent_min, video_latents.min().item())

    # 평균 및 분산 계산
    audio_mean = audio_latent_sum / num_audio_samples
    audio_variance = (audio_latent_square_sum / num_audio_samples) - (audio_mean**2)

    video_mean = video_latent_sum / num_video_samples
    video_variance = (video_latent_square_sum / num_video_samples) - (video_mean**2)

    # 결과 출력
    print("Audio Latent Statistics:")
    print(f"  Mean: {audio_mean}")
    print(f"  Variance: {audio_variance}")
    print(f"  Max: {audio_latent_max}")
    print(f"  Min: {audio_latent_min}")
    
    print("Video Latent Statistics:")
    print(f"  Mean: {video_mean}")
    print(f"  Variance: {video_variance}")
    print(f"  Max: {video_latent_max}")
    print(f"  Min: {video_latent_min}")

def main():
    # DataLoader 준비
    root_dir = "latents_data_32s_40frames"
    csv_file = os.path.join(root_dir, "dataset_info.csv")
    batch_size = 4
    gr_ac = 1
    dataloader = get_dataloader(root_dir, csv_file, batch_size=batch_size, shuffle=True, num_workers=0)

    # 모델 준비 및 학습 과정 생략
    # ...
    
    # 통계 계산
    compute_statistics(dataloader)

if __name__ == "__main__":
    main()
