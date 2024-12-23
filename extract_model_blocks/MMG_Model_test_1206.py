import torch
from diffusers import PNDMScheduler, UNet2DConditionModel
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
        device = cross_modal_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        dtype = cross_modal_config.get('dtype', torch.float32)
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
            ).to(dtype=dtype, device=device)
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
            ).to(dtype=dtype, device=device)
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
                print("    [down_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
                print("    [down_block][attention]")
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
                print("    [down_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
                output_states += (hidden_states,)
        return hidden_states, output_states


    def audio_mid_blocks(self, audio_unet, hidden_states, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs):
        if hasattr(audio_unet.mid_block, "has_cross_attention") and audio_unet.mid_block.has_cross_attention:
            print("  [mid_block][resnets]")
            hidden_states = audio_unet.mid_block.resnets[0](hidden_states, emb)
            for resnet, attn in zip(audio_unet.mid_block.resnets[1:], audio_unet.mid_block.attentions):
                print("  [mid_block][attention]")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                print("  [mid_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
        else:
            for resnet in audio_unet.mid_block.resnets:
                print("  [mid_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
        return hidden_states


    def audio_up_blocks(self, up_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, res_hidden_states_tuple):
        if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
            for resnet, attn in zip(up_block.resnets, up_block.attentions):
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
                print("    [up_block][resnet]")
                hidden_states = resnet(hidden_states, emb)

                print("    [up_block][attention]")
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
                print("    [up_block][resnet]")
                hidden_states = resnet(hidden_states, emb)
        return hidden_states


    def video_up_block(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
        print(f"  [video_output_blocks][{block_idx}]")
        h = torch.cat([h, hs.pop()], dim=1)
        for sublayer in video_unet.output_blocks[block_idx]:
            if isinstance(sublayer, Upsample):
                break
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        return h, hs


    def video_upsample(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
        print(f"  [video_Upsample][{block_idx}]")
        h = self.process_video_sublayer(video_unet.output_blocks[block_idx][-1], h, video_emb, video_context, batch_size)
        return h, hs


    def video_down_block(self, block_idx, video_unet, h, video_emb, video_context, batch_size, hs):
        print(f"  [video_input_blocks][{block_idx}]")
        for sublayer in video_unet.input_blocks[block_idx]:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        if block_idx == 0 and video_unet.addition_attention:
            h = video_unet.init_attn(h, video_emb, context=video_context, batch_size=batch_size)
        hs.append(h)
        return h, hs


    def process_video_sublayer(self, sublayer, h, video_emb, video_context, batch_size):
        layer_type = type(sublayer).__name__
        print(f"    [blocks]: {layer_type}")
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
        if self.audio_unet.config.center_input_sample:
            print("Centering input sample")
            audio_latents = 2 * audio_latents - 1.0
        print("Processing time embeddings")
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
        print("Processing encoder hidden states")
        if self.audio_unet.encoder_hid_proj is not None:
            audio_encoder_hidden_states = self.audio_unet.encoder_hid_proj(audio_encoder_hidden_states)
        print("Processing conv_in")
        audio_hidden_states = self.audio_unet.conv_in(audio_latents)
        audio_down_block_res_samples = (audio_hidden_states,)

        ###### Prepare Video ######
        video_emb = self.video_unet.time_embed(timestep_embedding(video_timestep, self.video_unet.model_channels))
        if self.video_unet.fps_cond:
            video_fps_tensor = torch.full_like(video_timestep, video_fps) if isinstance(video_fps, int) else video_fps
            video_emb += self.video_unet.fps_embedding(timestep_embedding(video_fps_tensor, self.video_unet.model_channels))
        b, _, t, _, _ = video_latents.shape
        video_context = video_context.repeat_interleave(repeats=t, dim=0) if video_context is not None else None
        video_emb = video_emb.repeat_interleave(repeats=t*b, dim=0)
        h = rearrange(video_latents, 'b c t h w -> (b t) c h w').type(self.video_unet.dtype)
        video_emb = video_emb.to(h.dtype)
        hs = []
        
        ####### Down Blocks ######
        print("Processing down_blocks")
        print("  [audio_down_block] 0")
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

        print("  [CrossModalTransformers] 0")
        cross_audio_latent = rearrange(audio_hidden_states, 'b c f (t tv) -> (b tv) c f t', tv=video_fps).contiguous()
        _h = self.video_cmt[0](h, cross_audio_latent)
        cross_audio_latent = self.audio_cmt[0](cross_audio_latent, h)
        audio_hidden_states = rearrange(cross_audio_latent, '(b tv) c f t -> b c f (t tv)', b=b, tv=video_fps).contiguous()
        h = _h
        ##################################################################################################################################

        print("    [audio_down_block][downsampler]")
        audio_hidden_states = self.audio_unet.down_blocks[0].downsamplers[0](audio_hidden_states)  # Audio downsample
        audio_down_block_res_samples += (audio_hidden_states,)
        h, hs = self.video_down_block(3, self.video_unet, h, video_emb, video_context, b, hs)  # Video downsample
    
        print("  [audio_down_block] 1")
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

        print("  [CrossModalTransformers] 1")
        cross_audio_latent = rearrange(audio_hidden_states, 'b c f (t tv) -> (b tv) c f t', tv=video_fps).contiguous()
        _h = self.video_cmt[1](h, cross_audio_latent)
        cross_audio_latent = self.audio_cmt[1](cross_audio_latent, h)
        audio_hidden_states = rearrange(cross_audio_latent, '(b tv) c f t -> b c f (t tv)', b=b, tv=video_fps).contiguous()
        h = _h
        ################################################################################################################################
        
        print("    [audio_down_block][downsampler]")
        audio_hidden_states = self.audio_unet.down_blocks[1].downsamplers[0](audio_hidden_states)  # Audio downsample
        audio_down_block_res_samples += (audio_hidden_states,)   
        h, hs = self.video_down_block(6, self.video_unet, h, video_emb, video_context, b, hs)  # Video downsample
        
        print("  [audio_down_block] 2")
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

        print("  [CrossModalTransformers] 2")
        cross_audio_latent = rearrange(audio_hidden_states, 'b c f (t tv) -> (b tv) c f t', tv=video_fps).contiguous()
        _h = self.video_cmt[2](h, cross_audio_latent)
        cross_audio_latent = self.audio_cmt[2](cross_audio_latent, h)
        audio_hidden_states = rearrange(cross_audio_latent, '(b tv) c f t -> b c f (t tv)', b=b, tv=video_fps).contiguous()
        h = _h
        ################################################################################################################################
        
        print("    [audio_down_block][downsampler]")
        audio_hidden_states = self.audio_unet.down_blocks[2].downsamplers[0](audio_hidden_states)  # Audio downsample
        audio_down_block_res_samples += (audio_hidden_states,)   
        h, hs = self.video_down_block(9, self.video_unet, h, video_emb, video_context, b, hs)  # Video downsample    
        
        print("  [audio_down_block] 3")
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
        print("Processing mid_block")
        audio_hidden_states = self.audio_mid_blocks(
            self.audio_unet, audio_hidden_states, audio_emb,
            audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs
        )
        for sublayer in self.video_unet.middle_block:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, b)


        ####### Up Blocks ######
        print("Processing up_blocks")
        print("  [audio_up_block] 0")
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

        print("  [CrossModalTransformers] 3")
        cross_audio_latent = rearrange(audio_hidden_states, 'b c f (t tv) -> (b tv) c f t', tv=video_fps).contiguous()
        _h = self.video_cmt[3](h, cross_audio_latent)
        cross_audio_latent = self.audio_cmt[3](cross_audio_latent, h)
        audio_hidden_states = rearrange(cross_audio_latent, '(b tv) c f t -> b c f (t tv)', b=b, tv=video_fps).contiguous()
        h = _h
        ################################################################################################################################

        print("    [audio_up_block][upsampler]")
        audio_hidden_states = self.audio_unet.up_blocks[0].upsamplers[0](audio_hidden_states)  # Audio upsample
        h, hs = self.video_upsample(2, self.video_unet, h, hs, video_emb, video_context, b)  # Video upsample

        print("  [audio_up_block] 1")
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

        print("  [CrossModalTransformers] 4")
        cross_audio_latent = rearrange(audio_hidden_states, 'b c f (t tv) -> (b tv) c f t', tv=video_fps).contiguous()
        _h = self.video_cmt[4](h, cross_audio_latent)
        cross_audio_latent = self.audio_cmt[4](cross_audio_latent, h)
        audio_hidden_states = rearrange(cross_audio_latent, '(b tv) c f t -> b c f (t tv)', b=b, tv=video_fps).contiguous()
        h = _h
        ################################################################################################################################

        print("    [audio_up_block][upsampler]")
        audio_hidden_states = self.audio_unet.up_blocks[1].upsamplers[0](audio_hidden_states)  # Audio upsample
        h, hs = self.video_upsample(5, self.video_unet, h, hs, video_emb, video_context, b)  # Video upsample

        print("  [audio_up_block] 2")
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

        print("  [CrossModalTransformers] 5")
        cross_audio_latent = rearrange(audio_hidden_states, 'b c f (t tv) -> (b tv) c f t', tv=video_fps).contiguous()
        _h = self.video_cmt[5](h, cross_audio_latent)
        cross_audio_latent = self.audio_cmt[5](cross_audio_latent, h)
        audio_hidden_states = rearrange(cross_audio_latent, '(b tv) c f t -> b c f (t tv)', b=b, tv=video_fps).contiguous()
        h = _h
        ################################################################################################################################

        print("    [audio_up_block][upsampler]")
        audio_hidden_states = self.audio_unet.up_blocks[2].upsamplers[0](audio_hidden_states)  # Audio upsample
        h, hs = self.video_upsample(8, self.video_unet, h, hs, video_emb, video_context, b)  # Video upsample

        print("  [audio_up_block] 3")
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
        print("Processing output layers")
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
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Prepare Audio UNet
    audio_unet = UNet2DConditionModel.from_pretrained("auffusion/auffusion-full", subfolder="unet").to(device, dtype)
    audio_unet.eval()

    # Prepare Video UNet
    video_config = OmegaConf.load('configs/inference_t2v_512_v2.0.yaml')
    video_model = instantiate_from_config(video_config.model)
    video_model.load_state_dict(torch.load('scripts/evaluation/model.ckpt')['state_dict'], strict=True)
    video_unet = video_model.model.diffusion_model.eval().to(device)

    # CrossModalTransformer Configuration
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device,
        'dtype': dtype
    }

    # Initialize the combined model
    model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config).to(device)

    # Prepare inputs
    audio_height, audio_width = 256, 1024
    audio_latents = torch.randn(
        (batch_size, audio_unet.config.in_channels, audio_height // 8, audio_width // 8),
        device=device,
        dtype=dtype,
    )
    scheduler = PNDMScheduler.from_pretrained("auffusion/auffusion", subfolder="scheduler")
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps, device=device)
    t = scheduler.timesteps[0]
    audio_encoder_hidden_states = torch.randn((batch_size, 77, audio_unet.config.cross_attention_dim), device=device, dtype=dtype)

    video_fps = 16
    video_latents = torch.randn(batch_size, 4, video_fps, 64, 64, device=device, dtype=dtype)
    video_timestep = torch.tensor([10], device=device)
    video_context = torch.randn(batch_size, 77, 1024, device=device, dtype=dtype)

    # Forward pass
    multi_modal_output = model(
        audio_latents=audio_latents,
        audio_timestep=t,
        audio_encoder_hidden_states=audio_encoder_hidden_states,
        video_latents=video_latents,
        video_timestep=video_timestep,
        video_context=video_context,
        video_fps=video_fps
    )


    # Compute original audio UNet output
    print("Computing original audio UNet output...")
    with torch.no_grad():
        audio_original_output = audio_unet(
            audio_latents,
            t,
            encoder_hidden_states=audio_encoder_hidden_states,
            return_dict=False,
        )[0]
    # Compute original video UNet output
    print("Computing original video UNet output...")
    with torch.no_grad():
        video_original_output = video_unet(video_latents, video_timestep, context=video_context, fps=video_fps)


    ######################### Compare Outputs #########################
    def compare_outputs(audio_output1: torch.Tensor, audio_output2: torch.Tensor, 
                       video_output1: torch.Tensor, video_output2: torch.Tensor, tol: float = 1e-5):
        """
        Compares two sets of audio and video output tensors.
        """
        if torch.equal(audio_output1, audio_output2):
            print("Audio outputs are exactly identical.")
        elif torch.allclose(audio_output1, audio_output2, atol=tol):
            print(f"Audio outputs are identical within tolerance {tol}.")
        else:
            print("Audio outputs differ.")
            diff = torch.abs(audio_output1 - audio_output2)
            print(f"Max audio difference: {diff.max().item()}")
            print(f"Mean audio difference: {diff.mean().item()}")
        
        if torch.equal(video_output1, video_output2):
            print("Video outputs are exactly identical.")
        elif torch.allclose(video_output1, video_output2, atol=tol):
            print(f"Video outputs are identical within tolerance {tol}.")
        else:
            print("Video outputs differ.")
            diff = torch.abs(video_output1 - video_output2)
            print(f"Max video difference: {diff.max().item()}")
            print(f"Mean video difference: {diff.mean().item()}")

    # Compare the outputs
    print("Comparing outputs...")
    audio_mm_output, video_mm_output = multi_modal_output
    compare_outputs(
        audio_output1=audio_original_output,
        audio_output2=audio_mm_output,
        video_output1=video_original_output,
        video_output2=video_mm_output
    )

if __name__ == "__main__":
    main()
