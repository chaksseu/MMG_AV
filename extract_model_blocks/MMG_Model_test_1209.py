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

from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn


"""
CrossModalTransformer Arguments:
- in_channels: Number of channels for the layer
- n_heads: Number of heads (channels // d_head, where d_head=64)
- d_head: 64
- depth: 1
- context_dim: Number of channels (same for both audio and video)
- use_checkpoint: True
- disable_self_attn: False
- use_linear: True
- img_cross_attention: False
"""

#################### CrossModalTransformer Blocks ##############################

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

audio_cmt = []
video_cmt = []
layer_channels = [320, 640, 1280, 1280, 1280, 640]

def initialize_basic_transformer_block(block):
    """
    Initializes the weights of a BasicTransformerBlock.
    """
    for name, param in block.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif param.dim() == 1:
            nn.init.zeros_(param)

def initialize_cross_modal_transformer(transformer):
    """
    Initializes the weights of a CrossModalTransformer.
    """
    if isinstance(transformer.proj_in, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(transformer.proj_in.weight)
        if transformer.proj_in.bias is not None:
            nn.init.zeros_(transformer.proj_in.bias)
    
    for block in transformer.transformer_blocks:
        initialize_basic_transformer_block(block)
    
    if isinstance(transformer.proj_out, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(transformer.proj_out.weight)
        if transformer.proj_out.bias is not None:
            nn.init.zeros_(transformer.proj_out.bias)


# Create and initialize CrossModalTransformers for audio and video
for channel in layer_channels:
    d_head = 64
    n_heads = channel // d_head
    transformer = CrossModalTransformer(
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
    
    # Initialize transformers
    initialize_cross_modal_transformer(transformer)
    
    # Append to respective lists
    audio_cmt.append(transformer)
    video_cmt.append(transformer)

###################################################################################


def audio_down_blocks(down_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, output_states):
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


def audio_mid_blocks(audio_unet, hidden_states, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs):
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


def audio_up_blocks(up_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, res_hidden_states_tuple):
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


def video_up_block(block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
    print(f"  [video_output_blocks][{block_idx}]")
    h = torch.cat([h, hs.pop()], dim=1)
    for sublayer in video_unet.output_blocks[block_idx]:
        if isinstance(sublayer, Upsample):
            break
        h = process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
    return h, hs


def video_upsample(block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
    print(f"  [video_Upsample][{block_idx}]")
    h = process_video_sublayer(video_unet.output_blocks[block_idx][-1], h, video_emb, video_context, batch_size)
    return h, hs


def video_down_block(block_idx, video_unet, h, video_emb, video_context, batch_size, hs):
    print(f"  [video_input_blocks][{block_idx}]")
    for sublayer in video_unet.input_blocks[block_idx]:
        h = process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
    if block_idx == 0 and video_unet.addition_attention:
        h = video_unet.init_attn(h, video_emb, context=video_context, batch_size=batch_size)
    hs.append(h)
    return h, hs


def process_video_sublayer(sublayer, h, video_emb, video_context, batch_size):
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


def mm_forward(audio_unet, audio_latents, audio_timestep, audio_encoder_hidden_states,
              video_unet, video_latents, video_timestep, video_context=None, video_fps=8,
              audio_attention_mask=None, audio_cross_attention_kwargs=None):
    ###### Prepare Audio ######
    if audio_unet.config.center_input_sample:
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
    audio_t_emb = audio_unet.time_proj(audio_timesteps).to(dtype=audio_latents.dtype)
    audio_emb = audio_unet.time_embedding(audio_t_emb)
    if audio_unet.time_embed_act is not None:
        audio_emb = audio_unet.time_embed_act(audio_emb)
    print("Processing encoder hidden states")
    if audio_unet.encoder_hid_proj is not None:
        audio_encoder_hidden_states = audio_unet.encoder_hid_proj(audio_encoder_hidden_states)
    print("Processing conv_in")
    audio_hidden_states = audio_unet.conv_in(audio_latents)
    audio_down_block_res_samples = (audio_hidden_states,)

    ###### Prepare Video ######
    video_emb = video_unet.time_embed(timestep_embedding(video_timestep, video_unet.model_channels))
    if video_unet.fps_cond:
        video_fps_tensor = torch.full_like(video_timestep, video_fps) if isinstance(video_fps, int) else video_fps
        video_emb += video_unet.fps_embedding(timestep_embedding(video_fps_tensor, video_unet.model_channels))
    b, _, t, _, _ = video_latents.shape
    video_context = video_context.repeat_interleave(repeats=t, dim=0) if video_context is not None else None
    video_emb = video_emb.repeat_interleave(repeats=t*b, dim=0)
    h = rearrange(video_latents, 'b c t h w -> (b t) c h w').type(video_unet.dtype)
    video_emb = video_emb.to(h.dtype)
    hs = []

    ####### Down Blocks ######
    print("Processing down_blocks")
    print("  [audio_down_block] 0")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(
        down_block=audio_unet.down_blocks[0],
        hidden_states=audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        emb=audio_emb,
        attention_mask=audio_attention_mask,
        cross_attention_kwargs=audio_cross_attention_kwargs,
        output_states=audio_down_block_res_samples
    )
    h, hs = video_down_block(0, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(1, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(2, video_unet, h, video_emb, video_context, b, hs)
    
    #################################################### Cross-Modal Transformer ###################################################

    print("  [CrossModalTransformers] 0")

    print("audio_hidden_states", audio_hidden_states.shape)
    print("h", h.shape)

    b_a, _, _, t_a = audio_hidden_states.shape
    b_v, _, h_v, w_v = h.shape
    k = int(b_v / (b_a * t_a))

    cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
    cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

    cross_video_latent_token = video_cmt[0](cross_video_latent_token, cross_audio_latent_token)
    cross_audio_latent_token = audio_cmt[0](cross_audio_latent_token, cross_video_latent_token)
    h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
    audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()

    ##################################################################################################################################

    print("    [audio_down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[0].downsamplers[0](audio_hidden_states)  # Audio downsample
    audio_down_block_res_samples += (audio_hidden_states,)
    h, hs = video_down_block(3, video_unet, h, video_emb, video_context, b, hs)  # Video downsample
   
    print("  [audio_down_block] 1")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(
        down_block=audio_unet.down_blocks[1],
        hidden_states=audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        emb=audio_emb,
        attention_mask=audio_attention_mask,
        cross_attention_kwargs=audio_cross_attention_kwargs,
        output_states=audio_down_block_res_samples
    )
    h, hs = video_down_block(4, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(5, video_unet, h, video_emb, video_context, b, hs)

    #################################################### Cross-Modal Transformer ###################################################

    print("  [CrossModalTransformers] 1")

    print("audio_hidden_states", audio_hidden_states.shape)
    print("h", h.shape)

    b_a, _, _, t_a = audio_hidden_states.shape
    b_v, _, h_v, w_v = h.shape
    k = int(b_v / (b_a * t_a))

    cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
    cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

    cross_video_latent_token = video_cmt[1](cross_video_latent_token, cross_audio_latent_token)
    cross_audio_latent_token = audio_cmt[1](cross_audio_latent_token, cross_video_latent_token)
    h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
    audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()

    ################################################################################################################################
    
    print("    [audio_down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[1].downsamplers[0](audio_hidden_states)  # Audio downsample
    audio_down_block_res_samples += (audio_hidden_states,)   
    h, hs = video_down_block(6, video_unet, h, video_emb, video_context, b, hs)  # Video downsample
    
    print("  [audio_down_block] 2")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(
        down_block=audio_unet.down_blocks[2],
        hidden_states=audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        emb=audio_emb,
        attention_mask=audio_attention_mask,
        cross_attention_kwargs=audio_cross_attention_kwargs,
        output_states=audio_down_block_res_samples
    )
    h, hs = video_down_block(7, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(8, video_unet, h, video_emb, video_context, b, hs)
    
    #################################################### Cross-Modal Transformer ###################################################

    print("  [CrossModalTransformers] 2")

    print("audio_hidden_states", audio_hidden_states.shape)
    print("h", h.shape)

    b_a, _, _, t_a = audio_hidden_states.shape
    b_v, _, h_v, w_v = h.shape
    k = int(b_v / (b_a * t_a))

    cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
    cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

    cross_video_latent_token = video_cmt[2](cross_video_latent_token, cross_audio_latent_token)
    cross_audio_latent_token = audio_cmt[2](cross_audio_latent_token, cross_video_latent_token)
    h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
    audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
 
    ################################################################################################################################
    
    print("    [audio_down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[2].downsamplers[0](audio_hidden_states)  # Audio downsample
    audio_down_block_res_samples += (audio_hidden_states,)   
    h, hs = video_down_block(9, video_unet, h, video_emb, video_context, b, hs)  # Video downsample    
    
    print("  [audio_down_block] 3")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(
        down_block=audio_unet.down_blocks[3],
        hidden_states=audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        emb=audio_emb,
        attention_mask=audio_attention_mask,
        cross_attention_kwargs=audio_cross_attention_kwargs,
        output_states=audio_down_block_res_samples
    )
    h, hs = video_down_block(10, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(11, video_unet, h, video_emb, video_context, b, hs)


    ####### Mid Block ######
    print("Processing mid_block")
    audio_hidden_states = audio_mid_blocks(
        audio_unet, audio_hidden_states, audio_emb,
        audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs
    )
    for sublayer in video_unet.middle_block:
        h = process_video_sublayer(sublayer, h, video_emb, video_context, b)


    ####### Up Blocks ######
    print("Processing up_blocks")
    print("  [audio_up_block] 0")
    if not audio_down_block_res_samples:
        raise ValueError("No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[0].resnets):]
    audio_down_block_res_samples = audio_down_block_res_samples[:-len(audio_unet.up_blocks[0].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(
        audio_unet.up_blocks[0],
        hidden_states=audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        emb=audio_emb,
        attention_mask=audio_attention_mask,
        cross_attention_kwargs=audio_cross_attention_kwargs,
        res_hidden_states_tuple=audio_res_hidden_states_tuple
    )
    h, hs = video_up_block(0, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(1, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(2, video_unet, h, hs, video_emb, video_context, b)  # R,U -> R

    #################################################### Cross-Modal Transformer ###################################################

    print("  [CrossModalTransformers] 3")

    print("audio_hidden_states", audio_hidden_states.shape)
    print("h", h.shape)

    b_a, _, _, t_a = audio_hidden_states.shape
    b_v, _, h_v, w_v = h.shape
    k = int(b_v / (b_a * t_a))

    cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
    cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

    cross_video_latent_token = video_cmt[3](cross_video_latent_token, cross_audio_latent_token)
    cross_audio_latent_token = audio_cmt[3](cross_audio_latent_token, cross_video_latent_token)
    h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
    audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()

    ################################################################################################################################

    print("    [audio_up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[0].upsamplers[0](audio_hidden_states)  # Audio upsample
    h, hs = video_upsample(2, video_unet, h, hs, video_emb, video_context, b)  # Video upsample

    print("  [audio_up_block] 1")
    if not audio_down_block_res_samples:
        raise ValueError("No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[1].resnets):]
    audio_down_block_res_samples = audio_down_block_res_samples[:-len(audio_unet.up_blocks[1].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(
        audio_unet.up_blocks[1],
        hidden_states=audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        emb=audio_emb,
        attention_mask=audio_attention_mask,
        cross_attention_kwargs=audio_cross_attention_kwargs,
        res_hidden_states_tuple=audio_res_hidden_states_tuple
    )
    h, hs = video_up_block(3, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(4, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(5, video_unet, h, hs, video_emb, video_context, b)  # R,S,T,U -> R,S,T
    
    #################################################### Cross-Modal Transformer ###################################################

    print("  [CrossModalTransformers] 4")

    print("audio_hidden_states", audio_hidden_states.shape)
    print("h", h.shape)

    b_a, _, _, t_a = audio_hidden_states.shape
    b_v, _, h_v, w_v = h.shape
    k = int(b_v / (b_a * t_a))

    cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
    cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

    cross_video_latent_token = video_cmt[4](cross_video_latent_token, cross_audio_latent_token)
    cross_audio_latent_token = audio_cmt[4](cross_audio_latent_token, cross_video_latent_token)
    h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
    audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()

    ################################################################################################################################

    print("    [audio_up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[1].upsamplers[0](audio_hidden_states)  # Audio upsample
    h, hs = video_upsample(5, video_unet, h, hs, video_emb, video_context, b)  # Video upsample

    print("  [audio_up_block] 2")
    if not audio_down_block_res_samples:
        raise ValueError("No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[2].resnets):]
    audio_down_block_res_samples = audio_down_block_res_samples[:-len(audio_unet.up_blocks[2].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(
        audio_unet.up_blocks[2],
        hidden_states=audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        emb=audio_emb,
        attention_mask=audio_attention_mask,
        cross_attention_kwargs=audio_cross_attention_kwargs,
        res_hidden_states_tuple=audio_res_hidden_states_tuple
    )
    h, hs = video_up_block(6, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(7, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(8, video_unet, h, hs, video_emb, video_context, b)  # R,S,T,U -> R,S,T
    
    #################################################### Cross-Modal Transformer ###################################################

    print("  [CrossModalTransformers] 5")

    print("audio_hidden_states", audio_hidden_states.shape)
    print("h", h.shape)

    b_a, _, _, t_a = audio_hidden_states.shape
    b_v, _, h_v, w_v = h.shape
    k = int(b_v / (b_a * t_a))

    cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k = k).contiguous()
    cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f').contiguous()

    cross_video_latent_token = video_cmt[5](cross_video_latent_token, cross_audio_latent_token)
    cross_audio_latent_token = audio_cmt[5](cross_audio_latent_token, cross_video_latent_token)
    h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k).contiguous()
    audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a).contiguous()
 
    ################################################################################################################################

    print("    [audio_up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[2].upsamplers[0](audio_hidden_states)  # Audio upsample
    h, hs = video_upsample(8, video_unet, h, hs, video_emb, video_context, b)  # Video upsample

    print("  [audio_up_block] 3")
    if not audio_down_block_res_samples:
        raise ValueError("No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[3].resnets):]
    audio_down_block_res_samples = audio_down_block_res_samples[:-len(audio_unet.up_blocks[3].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(
        audio_unet.up_blocks[3],
        hidden_states=audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        emb=audio_emb,
        attention_mask=audio_attention_mask,
        cross_attention_kwargs=audio_cross_attention_kwargs,
        res_hidden_states_tuple=audio_res_hidden_states_tuple
    )
    h, hs = video_up_block(9, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(10, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(11, video_unet, h, hs, video_emb, video_context, b)


    ###### Output Layers ######
    print("Processing output layers")
    if audio_unet.conv_norm_out is not None:
        audio_hidden_states = audio_unet.conv_norm_out(audio_hidden_states)
        if audio_unet.conv_act is not None:
            audio_hidden_states = audio_unet.conv_act(audio_hidden_states)
    audio_hidden_states = audio_unet.conv_out(audio_hidden_states)
    
    for sublayer in video_unet.out:
        h = sublayer(h)
    h = rearrange(h, '(b f) c h w -> b c f h w', b=b)

    audio_output, video_output = audio_hidden_states, h

    return audio_output, video_output


def main():
    """
    Main function: Loads models, prepares inputs, runs the multi-modal UNet forward pass,
    and compares the outputs with the original models.
    """
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    ############################# Prepare Audio #############################

    audio_unet = UNet2DConditionModel.from_pretrained("auffusion/auffusion-full", subfolder="unet").to(device, dtype)
    audio_unet.eval()
    
    # Prepare audio input data
    audio_height = 256  # Adjust as needed
    audio_width = 320  # Adjust as needed
    generator = torch.Generator(device=device).manual_seed(42)
    audio_encoder_hidden_states = torch.randn((batch_size, 77, audio_unet.config.cross_attention_dim), device=device, dtype=dtype)

    audio_latents = torch.randn(
        (batch_size, audio_unet.config.in_channels, audio_height // 8, audio_width // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # Set audio timesteps
    scheduler = PNDMScheduler.from_pretrained("auffusion/auffusion", subfolder="scheduler")
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    t = timesteps[0]

    # Compute original audio UNet output
    print("Computing original audio UNet output...")
    with torch.no_grad():
        audio_original_output = audio_unet(
            audio_latents,
            t,
            encoder_hidden_states=audio_encoder_hidden_states,
            return_dict=False,
        )[0]

    
    ############################# Prepare Video #############################

    video_config_path = 'configs/inference_t2v_512_v2.0.yaml'  # Update to actual path
    video_ckpt_path = 'scripts/evaluation/model.ckpt'  # Update to actual path

    video_config = OmegaConf.load(video_config_path)
    video_model = instantiate_from_config(video_config.model)
    video_model.load_state_dict(torch.load(video_ckpt_path)['state_dict'], strict=False)
    video_unet = video_model.model.diffusion_model.eval()

    # Generate input data
    video_fps = 40
    video_latents = torch.randn(1, 4, video_fps, 64, 64)  # (B, C, T, H, W)
    video_timestep = torch.tensor([10])
    video_context = torch.randn(1, 77, 1024)  # Example: text embeddings

    # Move models and tensors to device
    video_unet = video_unet.to(device)
    video_latents = video_latents.to(device)
    video_timestep = video_timestep.to(device)
    video_context = video_context.to(device)

    # Compute original video UNet output
    print("Computing original video UNet output...")
    with torch.no_grad():
        video_original_output = video_unet(video_latents, video_timestep, context=video_context, fps=video_fps)


    ######################## Multi-Modal UNet Forward Pass ########################

    print("Computing multi-modal UNet output...")
    with torch.no_grad():
        multi_modal_output = mm_forward(
            audio_unet=audio_unet,
            audio_latents=audio_latents,
            audio_timestep=t,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            video_unet=video_unet,
            video_latents=video_latents,
            video_timestep=video_timestep,
            video_context=video_context,
            video_fps=video_fps
        )


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
