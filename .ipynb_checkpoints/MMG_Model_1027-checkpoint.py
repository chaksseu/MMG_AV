import torch
from diffusers import PNDMScheduler, UNet2DConditionModel

from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer,
    Downsample, Upsample, TimestepBlock
)
from lvdm.common import checkpoint
from einops import rearrange
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config



def audio_down_blocks(down_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, output_states):
    if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
         # CrossAttention 있는 경우
        for j, (resnet, attn) in enumerate(zip(down_block.resnets, down_block.attentions)):
            print(f"    [down_block][resnet]")
            hidden_states = resnet(hidden_states, emb)
            print(f"    [down_block][attention]")
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False
            )[0]
            output_states = output_states + (hidden_states,)
    else:
        # CrossAttention 없는 경우
        for j, resnet in enumerate(down_block.resnets):
            print(f"    [down_block][resnet]")
            hidden_states = resnet(hidden_states, emb)
            output_states = output_states + (hidden_states,)
    return hidden_states, output_states


def audio_mid_blocks(audio_unet, audio_hidden_states, audio_emb, audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs):
    if hasattr(audio_unet.mid_block, "has_cross_attention") and audio_unet.mid_block.has_cross_attention:
        print(f"  [mid_block][resnets]")
        audio_hidden_states = audio_unet.mid_block.resnets[0](audio_hidden_states, audio_emb)
        for j, (resnet, attn) in enumerate(zip(audio_unet.mid_block.resnets[1:], audio_unet.mid_block.attentions)):
            print(f"  [mid_block][attention]")
            audio_hidden_states = attn(
                audio_hidden_states,
                encoder_hidden_states=audio_encoder_hidden_states,
                attention_mask=audio_attention_mask,
                cross_attention_kwargs=audio_cross_attention_kwargs,
                return_dict=False,
            )[0]
            print(f"  [mid_block][resnet]")
            audio_hidden_states = resnet(audio_hidden_states, audio_emb)
    else:
        for j, resnet in enumerate(audio_unet.mid_block.resnets):
            print(f"  [mid_block][resnet]")
            audio_hidden_states = resnet(audio_hidden_states, audio_emb)
    return audio_hidden_states


def audio_up_blocks(up_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, res_hidden_states_tuple):
    if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
        for j, (resnet, attn) in enumerate(zip(up_block.resnets, up_block.attentions)):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)        
            print(f"    [up_block][resnet]")
            hidden_states = resnet(hidden_states, emb)

            print(f"    [up_block][attention]")
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
    else:
        for j, resnet in enumerate(up_block.resnets):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)        
            print(f"    [up_block][resnet]")
            hidden_states = resnet(hidden_states, emb)
    return hidden_states

def video_up_block(block_idx, video_unet, h, hs, video_emb, video_context, b):
    print(f"  [video_output_blocks][{block_idx}]")
    h = torch.cat([h, hs.pop()], dim=1)
    for sublayer in video_unet.output_blocks[block_idx]:
        if type(sublayer).__name__ == "Upsample":
            break
        h = video_unet_process(sublayer, h, video_emb, video_context, b)
    return h, hs

def video_upsample(block_idx, video_unet, h, hs, video_emb, video_context, b):
    print(f"  [video_Upsample][{block_idx}]")
    h = video_unet_process(video_unet.output_blocks[block_idx][-1], h, video_emb, video_context, b)
    return h, hs

def video_down_block(block_idx, video_unet, h, video_emb, video_context, b, hs):
    print(f"  [video_input_blocks][{block_idx}]")
    for sublayer in video_unet.input_blocks[block_idx]:
        h = video_unet_process(sublayer, h, video_emb, video_context, b)
    if block_idx == 0 and video_unet.addition_attention:
        h = video_unet.init_attn(h, video_emb, context=video_context, batch_size=b)
    hs.append(h)
    return h, hs

def video_unet_process(sublayer, h, video_emb, video_context, b):
    layer_type = type(sublayer).__name__
    print(f"    [blocks]: {layer_type}")
    if isinstance(sublayer, TimestepBlock):
        h = sublayer(h, video_emb, batch_size=b)
    elif isinstance(sublayer, SpatialTransformer):
        h = sublayer(h, video_context)
    elif isinstance(sublayer, TemporalTransformer):
        h = rearrange(h, '(b f) c h w -> b c f h w', b=b)
        h = sublayer(h, video_context)
        h = rearrange(h, 'b c f h w -> (b f) c h w')
    else:
        #if isinstance(sublayer, Upsample):
        #    print("Upsample")
        #if isinstance(sublayer, Downsample):
        #    print("Downsample")        
        h = sublayer(h)
    return h



def mm_forward(audio_unet, audio_latents, audio_timestep, audio_encoder_hidden_states, video_unet, video_latents, video_timestep, video_context=None, video_fps=8, audio_attention_mask=None, audio_cross_attention_kwargs=None):

    ###### audio ready ######
    if audio_unet.config.center_input_sample:
        print("Centering input sample")
        audio_latents = 2 * audio_latents - 1.0
    print("Processing time embeddings")
    audio_timesteps = audio_timestep
    if not torch.is_tensor(audio_timesteps):
        is_mps = audio_latents.device.type == 'mps'
        if isinstance(audio_timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        audio_timesteps = torch.tensor([audio_timestep], dtype=dtype, device=audio_latents.device)
    elif len(audio_timesteps.shape) == 0:
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

    ###### video ready ######
    # 시간 및 임베딩 처리
    video_emb = video_unet.time_embed(timestep_embedding(video_timestep, video_unet.model_channels))
    # FPS 조건 처리
    if video_unet.fps_cond:
        video_fps = torch.full_like(video_timestep, video_fps) if isinstance(video_fps, int) else video_fps
        video_emb += video_unet.fps_embedding(timestep_embedding(video_fps, video_unet.model_channels))
    # context와 emb를 (b*t) 크기에 맞게 확장
    b, _, t, _, _ = video_latents.shape
    video_context = video_context.repeat_interleave(repeats=t, dim=0) if video_context is not None else None
    video_emb = video_emb.repeat_interleave(repeats=t, dim=0)
    # 입력 텐서 변환 및 데이터 타입 일치
    h = rearrange(video_latents, 'b c t h w -> (b t) c h w').type(video_unet.dtype)
    video_emb = video_emb.to(h.dtype)
    hs = []


    ####### down blocks ######
    print("Processing down_blocks")
    print(f"  [audio_down_block] 0")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(down_block= audio_unet.down_blocks[0], hidden_states = audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, output_states=audio_down_block_res_samples)
    h, hs = video_down_block(0, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(1, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(2, video_unet, h, video_emb, video_context, b, hs)
    ### Audio CrossModal Transformer ###
    print("latent_audio_shape:", audio_hidden_states.shape)
    print("latent_video_shape:", h.shape)
    ### Video CrossModal Transformer ###
    print(f"    [audio_down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[0].downsamplers[0](audio_hidden_states) # audio_downsample
    audio_down_block_res_samples = audio_down_block_res_samples + (audio_hidden_states,) 
    h, hs = video_down_block(3, video_unet, h, video_emb, video_context, b, hs) # video_downsample
   
    print(f"  [audio_down_block] 1")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(down_block= audio_unet.down_blocks[1], hidden_states = audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, output_states=audio_down_block_res_samples)
    h, hs = video_down_block(4, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(5, video_unet, h, video_emb, video_context, b, hs)
    ### Audio CrossModal Transformer ###
    print("latent_audio_shape:", audio_hidden_states.shape)
    print("latent_video_shape:", h.shape)
    ### Video CrossModal Transformer ###
    print(f"    [audio_down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[1].downsamplers[0](audio_hidden_states) # audio_downsample
    audio_down_block_res_samples = audio_down_block_res_samples + (audio_hidden_states,)   
    h, hs = video_down_block(6, video_unet, h, video_emb, video_context, b, hs) # video_downsample
    
    print(f"  [audio_down_block] 2")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(down_block= audio_unet.down_blocks[2], hidden_states = audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, output_states=audio_down_block_res_samples)
    h, hs = video_down_block(7, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(8, video_unet, h, video_emb, video_context, b, hs)
    ### Audio CrossModal Transformer ###
    print("latent_audio_shape:", audio_hidden_states.shape)
    print("latent_video_shape:", h.shape)
    ### Video CrossModal Transformer ###    
    print(f"    [audio_down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[2].downsamplers[0](audio_hidden_states) # audio_downsample
    audio_down_block_res_samples = audio_down_block_res_samples + (audio_hidden_states,)   
    h, hs = video_down_block(9, video_unet, h, video_emb, video_context, b, hs) # video_downsample    
    
    print(f"  [audio_down_block] 3")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(down_block= audio_unet.down_blocks[3], hidden_states = audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, output_states=audio_down_block_res_samples)
    h, hs = video_down_block(10, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(11, video_unet, h, video_emb, video_context, b, hs)


    ####### mid block ######
    print("Processing mid_block")
    audio_hidden_states = audio_mid_blocks(audio_unet, audio_hidden_states, audio_emb, audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs)
    for sublayer in video_unet.middle_block:
        h = video_unet_process(sublayer, h, video_emb, video_context, b)


    ####### up blocks ######
    print("Processing audio_down_block")
    print(f"  [audio_up_block] 0")
    if len(audio_down_block_res_samples) == 0:
        raise ValueError(f"No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[0].resnets) :]
    audio_down_block_res_samples = audio_down_block_res_samples[: -len(audio_unet.up_blocks[0].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(audio_unet.up_blocks[0], hidden_states=audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, res_hidden_states_tuple=audio_res_hidden_states_tuple)
    h, hs = video_up_block(0, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(1, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(2, video_unet, h, hs, video_emb, video_context, b) # R,U -> R
    ### Audio CrossModal Transformer ###
    print("latent_audio_shape:", audio_hidden_states.shape)
    print("latent_video_shape:", h.shape)
    ### Video CrossModal Transformer ###
    print(f"    [audio_up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[0].upsamplers[0](audio_hidden_states) # audio_upsample
    h, hs = video_upsample(2, video_unet, h, hs, video_emb, video_context, b) # video_upsample

    print(f"  [audio_up_block] 1")
    if len(audio_down_block_res_samples) == 0:
        raise ValueError(f"No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[1].resnets) :]
    audio_down_block_res_samples = audio_down_block_res_samples[: -len(audio_unet.up_blocks[1].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(audio_unet.up_blocks[1], hidden_states=audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, res_hidden_states_tuple=audio_res_hidden_states_tuple)
    h, hs = video_up_block(3, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(4, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(5, video_unet, h, hs, video_emb, video_context, b) # R,S,T,U -> R,S,T
    ### Audio CrossModal Transformer ###
    print("latent_audio_shape:", audio_hidden_states.shape)
    print("latent_video_shape:", h.shape)
    ### Video CrossModal Transformer ###
    print(f"    [audio_up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[1].upsamplers[0](audio_hidden_states) # audio_upsample
    h, hs = video_upsample(5, video_unet, h, hs, video_emb, video_context, b) # video_upsample

    print(f"  [audio_up_block] 2")
    if len(audio_down_block_res_samples) == 0:
        raise ValueError(f"No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[2].resnets) :]
    audio_down_block_res_samples = audio_down_block_res_samples[: -len(audio_unet.up_blocks[2].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(audio_unet.up_blocks[2], hidden_states=audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, res_hidden_states_tuple=audio_res_hidden_states_tuple)
    h, hs = video_up_block(6, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(7, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(8, video_unet, h, hs, video_emb, video_context, b) # R,S,T,U -> R,S,T
    ### Audio CrossModal Transformer ###
    print("latent_audio_shape:", audio_hidden_states.shape)
    print("latent_video_shape:", h.shape)
    ### Video CrossModal Transformer ###
    print(f"    [audio_up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[2].upsamplers[0](audio_hidden_states) # audio_upsample
    h, hs = video_upsample(8, video_unet, h, hs, video_emb, video_context, b) # video_upsample

    print(f"  [audio_up_block] 3")
    if len(audio_down_block_res_samples) == 0:
        raise ValueError(f"No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[3].resnets) :]
    audio_down_block_res_samples = audio_down_block_res_samples[: -len(audio_unet.up_blocks[3].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(audio_unet.up_blocks[3], hidden_states=audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, res_hidden_states_tuple=audio_res_hidden_states_tuple)
    h, hs = video_up_block(9, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(10, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(11, video_unet, h, hs, video_emb, video_context, b)


    ###### out layers ######
    print("Processing out:")
    if audio_unet.conv_norm_out is not None:
        audio_hidden_states = audio_unet.conv_norm_out(audio_hidden_states)
        if audio_unet.conv_act is not None:
            audio_hidden_states = audio_unet.conv_act(audio_hidden_states)
    audio_hidden_states = audio_unet.conv_out(audio_hidden_states)
    for idx, sublayer in enumerate(video_unet.out):
        h = sublayer(h)
    h = rearrange(h, '(b f) c h w -> b c f h w', b=b)

    audio_unet_output, video_unet_output = audio_hidden_states, h
    return (audio_unet_output, video_unet_output)


def main():
    """
    메인 함수: 모델을 로드하고, 입력을 준비한 후,
    원본 모델과 레이어 추출 함수를 통해 출력을 비교합니다.
    """
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32


    ############################# audio 준비 #############################

    audio_unet = UNet2DConditionModel.from_pretrained("auffusion/auffusion-full", subfolder="unet").to(device, dtype)
    audio_unet.eval()
    
    # audio 입력 데이터 준비
    
    audio_height = 64  # 필요한 값으로 변경 가능
    audio_width = 128  # 필요한 값으로 변경 가능
    generator = torch.Generator(device=device).manual_seed(42)
    audio_encoder_hidden_states = torch.randn((batch_size, 77, audio_unet.config.cross_attention_dim), device=device, dtype=dtype)

    audio_latents = torch.randn(
        (batch_size, audio_unet.config.in_channels, audio_height // 8, audio_width // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # 오디오 타임스텝 설정
    scheduler = PNDMScheduler.from_pretrained("auffusion/auffusion", subfolder="scheduler")
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    t = timesteps[0]

    # 원본 audio UNet 출력 계산
    print("Computing original UNet output...")
    with torch.no_grad():
        audio_original_output = audio_unet(
            audio_latents,
            t,
            encoder_hidden_states=audio_encoder_hidden_states,
            return_dict=False,
        )[0]


    ############################# video 준비 #############################

    video_config_path = 'configs/inference_t2v_512_v2.0.yaml'  # 실제 경로로 수정하세요
    video_ckpt_path = 'scripts/evaluation/model.ckpt'  # 실제 경로로 수정하세요

    video_config = OmegaConf.load(video_config_path)
    video_model = instantiate_from_config(video_config.model)
    video_model.load_state_dict(torch.load(video_ckpt_path)['state_dict'], strict=False)
    video_unet = video_model.model.diffusion_model.eval()


    # 입력 데이터 생성
    video_latents = torch.randn(1, 4, 2, 16, 32)  # (B, C, T, H, W) => torch.randn(1, 4, 16, 64, 64)

    video_timestep = torch.tensor([10])
    video_context = torch.randn(1, 77, 1024)  # 예시로 텍스트 임베딩 사용
    video_fps = 8

    # 모델과 텐서를 디바이스로 이동
    video_unet = video_unet.to(device)
    video_latents = video_latents.to(device)
    video_timestep = video_timestep.to(device)
    video_context = video_context.to(device)
    #video_fps = video_fps.to(device)

    # 원본 video UNet 출력 계산
    video_original_output = video_unet(video_latents, video_timestep, context=video_context, fps=video_fps)


    ######################## multi modal unet 실행 ########################


    # mm_forward
    print("Computing extracted layers multi UNet output...")
    with torch.no_grad(): 
        extract_model_output = mm_forward(
            audio_unet = audio_unet,
            video_unet = video_unet,
            audio_latents = audio_latents,
            video_latents = video_latents,
            video_timestep = video_timestep,
            audio_timestep = t,
            audio_encoder_hidden_states = audio_encoder_hidden_states,
            video_context = video_context,
            video_fps = video_fps
        )




    ######################### 출력 비교 함수 #########################
    def compare_outputs(audio_output1: torch.Tensor, audio_output2: torch.Tensor, video_output1: torch.Tensor, video_output2: torch.Tensor,tol: float = 1e-5):
        """
        두 출력 텐서를 비교합니다.
        """
        if torch.equal(audio_output1, audio_output2):
            print("오디오의 두 출력이 완전히 동일합니다.")
        elif torch.allclose(audio_output1, audio_output2, atol=tol):
            print(f"오디오의 두 출력은 허용 오차 {tol} 이내에서 동일합니다.")
        else:
            print("오디오의 두 출력이 다릅니다.")
            diff = torch.abs(audio_output1 - audio_output2)
            print(f"오디오 최대 차이: {diff.max().item()}")
            print(f"오디오 평균 차이: {diff.mean().item()}")
        
        if torch.equal(video_output1, video_output2):
            print("비디오의 두 출력이 완전히 동일합니다.")
        elif torch.allclose(video_output1, video_output2, atol=tol):
            print(f"비디오의 두 출력은 허용 오차 {tol} 이내에서 동일합니다.")
        else:
            print("비디오의 두 출력이 다릅니다.")
            diff = torch.abs(video_output1 - video_output2)
            print(f"비디오 최대 차이: {diff.max().item()}")
            print(f"비디오 평균 차이: {diff.mean().item()}")

    # 두 출력 비교
    print("Comparing outputs...")
    audio_extract_model_output = extract_model_output[0]
    video_extract_model_output = extract_model_output[1]
    compare_outputs(audio_output1 = audio_original_output, video_output1 = video_original_output, audio_output2 = audio_extract_model_output, video_output2 = video_extract_model_output)


if __name__ == "__main__":
    main()
