import torch
from einops import rearrange
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer,
    Downsample, Upsample, TimestepBlock
)
from lvdm.common import checkpoint

def forward_with_extracted_layers(model, x, timesteps, context=None, fps=16):
    """
    각 레이어를 개별적으로 처리하면서 모델의 순전파 과정을 재현하고,
    최종 결과가 동일하도록 합니다.
    """
    diffusion_model = model.model.diffusion_model
    t_emb = timestep_embedding(timesteps, diffusion_model.model_channels)
    emb = diffusion_model.time_embed(t_emb)

    if diffusion_model.fps_cond:
        if isinstance(fps, int):
            fps = torch.full_like(timesteps, fps)
        fps_emb = timestep_embedding(fps, diffusion_model.model_channels)
        emb += diffusion_model.fps_embedding(fps_emb)

    b, _, t, _, _ = x.shape
    # context와 emb를 (b*t) 크기에 맞게 확장
    if context is not None:
        context = context.repeat_interleave(repeats=t, dim=0)
    emb = emb.repeat_interleave(repeats=t, dim=0)

    # 입력 텐서를 (B*T, C, H, W) 형태로 변환
    h = rearrange(x, 'b c t h w -> (b t) c h w')
    h = h.type(diffusion_model.dtype)
    emb = emb.to(h.dtype)

    hs = []

    # input_blocks 처리
    print("Processing input_blocks:")
    for block_idx, block in enumerate(diffusion_model.input_blocks):
        print(f"  [input_blocks][{block_idx}]")
        layer_idx = 0  # 레이어 인덱스 초기화
        for sublayer in block:
            layer_type = type(sublayer).__name__
            print(f"    [input_blocks][{block_idx}][{layer_idx}]: {layer_type}")
            if isinstance(sublayer, TimestepBlock):
                h = sublayer(h, emb, batch_size=b)
            elif isinstance(sublayer, SpatialTransformer):
                h = sublayer(h, context)
            elif isinstance(sublayer, TemporalTransformer):
                h = rearrange(h, '(b f) c h w -> b c f h w', b=b)
                h = sublayer(h, context)
                h = rearrange(h, 'b c f h w -> (b f) c h w')
            elif isinstance(sublayer, Downsample):
                h = sublayer(h)
            else:
                h = sublayer(h)
            layer_idx += 1
        if block_idx == 0 and diffusion_model.addition_attention:
            h = diffusion_model.init_attn(h, emb, context=context, batch_size=b)
        hs.append(h)

    # middle_block 처리
    print("Processing middle_block:")
    layer_idx = 0  # 레이어 인덱스 초기화
    for sublayer in diffusion_model.middle_block:
        layer_type = type(sublayer).__name__
        print(f"  [middle_block][{layer_idx}]: {layer_type}")
        if isinstance(sublayer, TimestepBlock):
            h = sublayer(h, emb, batch_size=b)
        elif isinstance(sublayer, SpatialTransformer):
            h = sublayer(h, context)
        elif isinstance(sublayer, TemporalTransformer):
            h = rearrange(h, '(b f) c h w -> b c f h w', b=b)
            h = sublayer(h, context)
            h = rearrange(h, 'b c f h w -> (b f) c h w')
        else:
            h = sublayer(h)
        layer_idx += 1

    # output_blocks 처리
    print("Processing output_blocks:")
    for block_idx, block in enumerate(diffusion_model.output_blocks):
        h_prev = hs.pop()
        h = torch.cat([h, h_prev], dim=1)
        print(f"  [output_blocks][{block_idx}]")
        layer_idx = 0  # 레이어 인덱스 초기화
        for sublayer in block:
            layer_type = type(sublayer).__name__
            print(f"    [output_blocks][{block_idx}][{layer_idx}]: {layer_type}")
            if isinstance(sublayer, TimestepBlock):
                h = sublayer(h, emb, batch_size=b)
            elif isinstance(sublayer, SpatialTransformer):
                h = sublayer(h, context)
            elif isinstance(sublayer, TemporalTransformer):
                h = rearrange(h, '(b f) c h w -> b c f h w', b=b)
                h = sublayer(h, context)
                h = rearrange(h, 'b c f h w -> (b f) c h w')
            elif isinstance(sublayer, Upsample):
                h = sublayer(h)
            else:
                h = sublayer(h)
            layer_idx += 1

    # out 처리
    print("Processing out:")
    layer_idx = 0  # 레이어 인덱스 초기화
    for sublayer in diffusion_model.out:
        layer_type = type(sublayer).__name__
        print(f"  [out][{layer_idx}]: {layer_type}")
        h = sublayer(h)
        layer_idx += 1

    # 텐서 형태 복원: (B*T, C, H, W) -> (B, C, T, H, W)
    h = rearrange(h, '(b f) c h w -> b c f h w', b=b)
    return h

# 사용 예시
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

config_path = 'configs/inference_t2v_512_v2.0.yaml'  # 실제 경로로 수정하세요
ckpt_path = 'scripts/evaluation/model.ckpt'  # 실제 경로로 수정하세요

config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
model = model.eval()

# 입력 데이터 생성
x = torch.randn(1, 4, 16, 64, 64)  # (B, C, T, H, W) => torch.randn(1, 4, 16, 64, 64)

timesteps = torch.tensor([10])
context = torch.randn(1, 77, 1024)  # 예시로 텍스트 임베딩 사용

# 함수 실행
original_output = model.model.diffusion_model(x, timesteps, context=context, fps=16)
extract_model_output = forward_with_extracted_layers(model, x, timesteps, context=context, fps=16)

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
compare_outputs(original_output, extract_model_output)






### 모델 구초 출력 코드 ###
'''
def print_block_info(block, block_type, i, j=None):
    """
    특정 블록의 타입과 내용을 출력하는 함수.
    block: 탐색할 블록 객체
    block_type: 'input', 'middle', 'output' 중 하나
    i, j: 해당 블록 위치 인덱스
    """
    if j is not None:
        print(f"[{block_type}_blocks][{i}][{j}]: {block.__class__.__name__}")
    else:
        print(f"[{block_type}_blocks][{i}]: {block.__class__.__name__}")
    #print(block)  # 해당 블록의 구조 출력
    #print("-" * 80)

# 모델의 특정 블록들을 탐색하는 코드
model_blocks = model.model.diffusion_model

print("-" * 80)
# 1. input_blocks 탐색
for i, block in enumerate(model_blocks.input_blocks):
    # 중첩된 블록 탐색
    for j, sub_block in enumerate(block):
        print_block_info(sub_block, "input", i, j)
print("-" * 80)
# 2. middle_block 탐색
for i, block in enumerate(model_blocks.middle_block):
    print_block_info(block, "middle", i)
print("-" * 80)
# 3. output_blocks 탐색
for i, block in enumerate(model_blocks.output_blocks):
    for j, sub_block in enumerate(block):
        print_block_info(sub_block, "output", i, j)
print("-" * 80)
'''