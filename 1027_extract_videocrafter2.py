import torch
from einops import rearrange
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer,
    Downsample, Upsample, TimestepBlock
)
from lvdm.common import checkpoint


def video_up_block(block_idx, video_unet, h, hs, video_emb, video_context, b):
    print(f"  [output_blocks][{block_idx}]")
    h = torch.cat([h, hs.pop()], dim=1)
    for sublayer in video_unet.output_blocks[block_idx]:
        if type(sublayer).__name__ == "Upsample":
            break
        h = video_unet_process(sublayer, h, video_emb, video_context, b)
    return h, hs

def video_upsample(block_idx, video_unet, h, hs, video_emb, video_context, b):
    print(f"  [Upsample][{block_idx}]")
    h = video_unet_process(video_unet.output_blocks[block_idx][-1], h, video_emb, video_context, b)
    return h, hs


def video_down_block(block_idx, video_unet, h, video_emb, video_context, b, hs):
    print(f"  [input_blocks][{block_idx}]")
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
        if isinstance(sublayer, Upsample):
            print("Upsample")
        if isinstance(sublayer, Downsample):
            print("Downsample")        
        h = sublayer(h)
    return h


def forward_with_extracted_layers(video_unet, video_latent, video_timesteps, video_context=None, video_fps=8):
    """
    각 레이어를 개별적으로 처리하면서 모델의 순전파 과정을 재현하고,
    최종 결과가 동일하도록 합니다.
    """

    # 시간 및 임베딩 처리
    video_emb = video_unet.time_embed(timestep_embedding(video_timesteps, video_unet.model_channels))
    # FPS 조건 처리
    if video_unet.fps_cond:
        video_fps = torch.full_like(video_timesteps, video_fps) if isinstance(video_fps, int) else video_fps
        video_emb += video_unet.fps_embedding(timestep_embedding(video_fps, video_unet.model_channels))
    # context와 emb를 (b*t) 크기에 맞게 확장
    b, _, t, _, _ = video_latent.shape
    video_context = video_context.repeat_interleave(repeats=t, dim=0) if video_context is not None else None
    video_emb = video_emb.repeat_interleave(repeats=t, dim=0)
    # 입력 텐서 변환 및 데이터 타입 일치
    h = rearrange(video_latent, 'b c t h w -> (b t) c h w').type(video_unet.dtype)
    video_emb = video_emb.to(h.dtype)
    hs = []

    # input_blocks 처리

    print("Processing input_blocks:")

    h, hs = video_down_block(0, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(1, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(2, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(3, video_unet, h, video_emb, video_context, b, hs) # downsample
    h, hs = video_down_block(4, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(5, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(6, video_unet, h, video_emb, video_context, b, hs) # downsample
    h, hs = video_down_block(7, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(8, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(9, video_unet, h, video_emb, video_context, b, hs) # downsample
    h, hs = video_down_block(10, video_unet, h, video_emb, video_context, b, hs)
    h, hs = video_down_block(11, video_unet, h, video_emb, video_context, b, hs)
    

    # middle_block 처리
    
    print("Processing middle_block:")
    for sublayer in video_unet.middle_block:
        h = video_unet_process(sublayer, h, video_emb, video_context, b)
    '''
    h = video_unet_process(video_unet.middle_block[0], h, video_emb, video_context, b)
    h = video_unet_process(video_unet.middle_block[1], h, video_emb, video_context, b)
    h = video_unet_process(video_unet.middle_block[2], h, video_emb, video_context, b)
    h = video_unet_process(video_unet.middle_block[3], h, video_emb, video_context, b)
    '''


    # output_blocks 처리 
    print("Processing output_blocks:")
    h, hs = video_up_block(0, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(1, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(2, video_unet, h, hs, video_emb, video_context, b) # R,U
    h, hs = video_upsample(2, video_unet, h, hs, video_emb, video_context, b)

    h, hs = video_up_block(3, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(4, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(5, video_unet, h, hs, video_emb, video_context, b) # R,S,T,U
    h, hs = video_upsample(5, video_unet, h, hs, video_emb, video_context, b)


    h, hs = video_up_block(6, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(7, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(8, video_unet, h, hs, video_emb, video_context, b) # R,S,T,U
    h, hs = video_upsample(8, video_unet, h, hs, video_emb, video_context, b)


    h, hs = video_up_block(9, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(10, video_unet, h, hs, video_emb, video_context, b)
    h, hs = video_up_block(11, video_unet, h, hs, video_emb, video_context, b)


    # out 처리
    print("Processing out:")
    for idx, sublayer in enumerate(video_unet.out):
        print(f"  [out][{idx}]: {type(sublayer).__name__}")
        h = sublayer(h)
    # 텐서 형태 복원
    h = rearrange(h, '(b f) c h w -> b c f h w', b=b)

    return h



# 사용 예시
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config




video_config_path = 'configs/inference_t2v_512_v2.0.yaml'  # 실제 경로로 수정하세요
video_ckpt_path = 'scripts/evaluation/model.ckpt'  # 실제 경로로 수정하세요

video_config = OmegaConf.load(video_config_path)
video_model = instantiate_from_config(video_config.model)
video_model.load_state_dict(torch.load(video_ckpt_path)['state_dict'], strict=False)
video_unet = video_model.model.diffusion_model.eval()


# 입력 데이터 생성
video_latent = torch.randn(1, 4, 1, 16, 16)  # (B, C, T, H, W) => torch.randn(1, 4, 16, 64, 64)

video_timestep = torch.tensor([10])
video_context = torch.randn(1, 77, 1024)  # 예시로 텍스트 임베딩 사용
video_fps = 8



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 텐서를 디바이스로 이동
video_unet = video_unet.to(device)
video_latent = video_latent.to(device)
video_timestep = video_timestep.to(device)
video_context = video_context.to(device)
#video_fps = video_fps.to(device)

# 원본 video UNet 출력 계산
video_original_output = video_unet(video_latent, video_timestep, context=video_context, fps=video_fps)

video_extract_model_output = forward_with_extracted_layers(video_unet, video_latent, video_timestep, video_context=video_context, video_fps=video_fps)



def compare_outputs(output1, output2, tol=1e-5):
    """
    두 출력 텐서를 비교합니다.
    """
    if torch.equal(output1, output2):
        print("두 출력이 완전히 동일합니다.")
    elif torch.allclose(output1, output2, atol=tol):
        print(f"두 출력은 허용 오차 {tol} 이내에서 동일합니다.")
    else:
        print("두 출력이 다릅니다.")
        diff = torch.abs(output1 - output2)
        print(f"최대 차이: {diff.max().item()}")
        print(f"평균 차이: {diff.mean().item()}")

# 두 출력 비교
compare_outputs(video_original_output, video_extract_model_output)


