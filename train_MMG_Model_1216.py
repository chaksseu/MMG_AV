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



os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ['TORCH_USE_CUDA_DSA'] = "0"

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


        # 추가: 어댑테이션을 위한 linear layer
        # audio용, video용 각각 준비
        self.audio_adaptation_in = nn.ModuleList()
        self.video_adaptation_in = nn.ModuleList()

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

            # Adaptation layers
            self.audio_adaptation_in.append(nn.Linear(channel, channel))
            self.video_adaptation_in.append(nn.Linear(channel, channel))

        self.initialize_linear_layers()



    def initialize_linear_layers(self):
        """
        Initializes the weights and biases of all adaptation Linear layers.
        """
        for linear in self.audio_adaptation_in:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        '''
        for linear in self.audio_adaptation_out:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        '''
        for linear in self.video_adaptation_in:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        '''
        for linear in self.video_adaptation_out:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        '''
        
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
    

    def linear_cmt(self, audio_hidden_states, h, index):
        
        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

        # --- Linear In ---
        
        condition_cross_audio_latent_token = cross_audio_latent_token
        condition_cross_video_latent_token = cross_video_latent_token
        
        condition_cross_video_latent_token = rearrange(condition_cross_video_latent_token, 'b c n -> (b n) c')
        condition_cross_video_latent_token = self.video_adaptation_in[index](condition_cross_video_latent_token)
        condition_cross_video_latent_token = rearrange(condition_cross_video_latent_token, '(b n) c -> b c n', b=b_a*t_a)
        condition_cross_audio_latent_token = rearrange(condition_cross_audio_latent_token, 'bt c f -> (bt f) c')
        condition_cross_audio_latent_token = self.audio_adaptation_in[index](condition_cross_audio_latent_token)
        condition_cross_audio_latent_token = rearrange(condition_cross_audio_latent_token, '(bt f) c -> bt c f', bt=b_a*t_a)
        
        # --- Cross Modal Transformer ---
        cross_video_latent_token = self.video_cmt[index](cross_video_latent_token, condition_cross_audio_latent_token)        
        cross_audio_latent_token = self.audio_cmt[index](cross_audio_latent_token, condition_cross_video_latent_token)
        
        
        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
        
        return audio_hidden_states, h
    

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
        
        audio_hidden_states , h = self.linear_cmt(audio_hidden_states, h, 0)
        
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
        
        audio_hidden_states , h = self.linear_cmt(audio_hidden_states, h, 1)
       
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
      
        audio_hidden_states , h = self.linear_cmt(audio_hidden_states, h, 2)
        
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
   
     #   print("  [CrossModalTransformers] 3")
        
        audio_hidden_states , h = self.linear_cmt(audio_hidden_states, h, 3)

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

        audio_hidden_states , h = self.linear_cmt(audio_hidden_states, h, 4)

        ##############################################################################################################################

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

        audio_hidden_states , h = self.linear_cmt(audio_hidden_states, h, 5)

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


def main():
    # DataLoader 준비
    root_dir = "latents_data_32s_40frames_vggsound_sparse_new_normalization"
    csv_file = os.path.join(root_dir, "dataset_info.csv")
    dataset_name="vggsound_sparse"
    batch_size = 4
    gr_ac = 32
    lr = 1e-6
    full_batch_size = batch_size * 8 * gr_ac
    video_fps = 12.5
    date = "1217_new_linear"
    video_fps = torch.tensor([video_fps]*batch_size).long()


    dataloader = get_dataloader(root_dir, csv_file, batch_size=batch_size, shuffle=True, num_workers=0)

    # Accelerate 초기화
    accelerator = Accelerator(
        mixed_precision="bf16",  # 혹은 bf16
        gradient_accumulation_steps=gr_ac  # 예: 그래디언트 누적
    )
    device = accelerator.device

    # Audio UNet 준비
    audio_unet = UNet2DConditionModel.from_pretrained("auffusion/auffusion-full", subfolder="unet")
    audio_unet.eval()

    # Video UNet 준비
    video_config = OmegaConf.load('configs/inference_t2v_512_v2.0.yaml')
    video_model = instantiate_from_config(video_config.model)
    state_dict = torch.load('scripts/evaluation/model.ckpt')['state_dict']
    video_model.load_state_dict(state_dict, strict=True)
    video_model.to(device)
    video_unet = video_model.model.diffusion_model.eval()

    # CrossModal Config
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device
    }


    # Initialize the combined model
    model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config)

    
    #noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    noise_scheduler = DDPMScheduler.from_pretrained("auffusion/auffusion", subfolder="scheduler")

    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())

    # 학습 가능한 파라미터 수 계산
    trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))

    print(f"전체 파라미터 수: {total_params}")
    print(f"학습 가능한 파라미터 수: {trainable_params}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()


    #audio_null_text_embed = torch.load("null_text_embedding/audio_text_embeds/audio_text_embed_null.pt")
    #audio_null_text_embed = audio_null_text_embed.unsqueeze(0).expand(batch_size, -1, -1)
  
    #video_null_text_embed = torch.load("null_text_embedding/video_text_embeds/video_text_embed_null.pt")
    #video_null_text_embed = video_null_text_embed.unsqueeze(0).expand(batch_size, -1, -1)

    num_epochs = 1000
    global_step = 0
    losses = []
    losses_video = []
    losses_audio = []

    if accelerator.is_main_process:
        wandb.init(project="my_diffusion_project", name="train_run")
    else:
        os.environ["WANDB_MODE"] = "offline"
    
    for epoch in range(num_epochs):
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # batch에서 데이터 추출
                audio_latents = batch["audio_latent"]
                video_latents = batch["video_latent"]
                audio_text_embed = batch["audio_text_embed"]
                video_text_embed = batch["video_text_embed"]



                #print("audio_text_embed.shape",audio_text_embed.shape)
                #print("video_text_embed.shape",video_text_embed.shape)

                #print("audio range", audio_latents.min(), audio_latents.max())
                #print("video range", video_latents.min(), video_latents.max())
                #print(f"Audio Input: {audio_latents.mean().item()}, {audio_latents.std().item()}")
                #print(f"Video Input: {video_latents.mean().item()}, {video_latents.std().item()}")



                # cfg를 위한 null conditioning
                # null 임베딩 (모든 샘플에 대해 동일하게 사용할 경우)
                audio_null_text_embed = torch.zeros(1, 1, audio_text_embed.shape[-1], device=audio_text_embed.device)  # Shape: (1, 1, embed_dim)
                video_null_text_embed = torch.zeros(1, 1, video_text_embed.shape[-1], device=video_text_embed.device)  # Shape: (1, 1, embed_dim)

                # 배치 크기에 맞게 null 임베딩 확장 (브로드캐스팅을 활용)
                # 별도의 확장이 필요하지 않음, 브로드캐스팅을 통해 자동으로 맞춰짐

                # 10% 확률로 각 샘플마다 임베딩을 null 임베딩으로 대체
                mask = (torch.rand(batch_size, 1, 1, device=audio_text_embed.device) < 0.1)  # Shape: (batch_size, 1, 1)

                # torch.where를 사용해 null 임베딩 적용 (브로드캐스팅을 통해 자동 확장)
                audio_text_embed = torch.where(mask, audio_null_text_embed, audio_text_embed)
                video_text_embed = torch.where(mask, video_null_text_embed, video_text_embed)



                
                # 랜덤 t 선택
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()
                #video_ts = torch.full((batch_size,), timesteps[0], device=device, dtype=torch.long)
                
                noise_audio = torch.randn_like(audio_latents).to(device)
                noise_video = torch.randn_like(video_latents).to(device)

                
                noised_audio_latents = noise_scheduler.add_noise(audio_latents, noise_audio, timesteps)
                noised_video_latents = video_model.q_sample(x_start=video_latents, t=timesteps, noise=noise_video)


                # Forward
                audio_output, video_output = model(
                    audio_latents=noised_audio_latents,
                    audio_timestep=timesteps,
                    audio_encoder_hidden_states=audio_text_embed,
                    video_latents=noised_video_latents,
                    video_timestep=timesteps,
                    video_context=video_text_embed,
                    video_fps=video_fps
                )

                #print(f"Audio Output: {audio_output.mean().item()}, {audio_output.std().item()}")
                #print(f"Noise Audio: {noise_audio.mean().item()}, {noise_audio.std().item()}")
                #print(f"Video Output: {video_output.mean().item()}, {video_output.std().item()}")
                #print(f"Noise Video: {noise_video.mean().item()}, {noise_video.std().item()}")

                audio_weight, video_weight = 1.0, 1.0 
                loss_audio = audio_weight * F.mse_loss(audio_output, noise_audio)
                loss_video = video_weight * F.mse_loss(video_output, noise_video)
                loss = loss_audio + loss_video

                losses.append(loss.item())
                losses_audio.append(loss_audio.item())
                losses_video.append(loss_video.item())


                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                #tepoch.set_postfix(loss_video=loss_video.item(), loss_audio=loss_audio.item())
                
                

                if accelerator.is_main_process and batch_idx % gr_ac == 0:
                    avg_losses = sum(losses) / len(losses)
                    losses = []  # 초기화
                    avg_losses_video = sum(losses_video) / len(losses_video)
                    losses_video = []  # 초기화
                    avg_losses_audio = sum(losses_audio) / len(losses_audio)
                    losses_audio = []  # 초기화

                    wandb.log({
                        "train/loss": avg_losses,
                        "train/loss_audio": avg_losses_audio,
                        "train/loss_video": avg_losses_video,
                        "epoch": epoch,
                        "step": global_step
                    })
                
                if accelerator.is_main_process and global_step % 1000 == 0:
                    checkpoint_path = f"mmg_checkpoints/{date}_lr_{lr}_batch_{full_batch_size}_global_step_{global_step}_{dataset_name}"
                    accelerator.save_state(checkpoint_path)

    print("Training Done.")


    

if __name__ == "__main__":
    main()
