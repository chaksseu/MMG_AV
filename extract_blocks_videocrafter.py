import torch
from collections import OrderedDict
from einops import rearrange

# 필요한 모듈 임포트
from lvdm.models.ddpm3d import LatentDiffusion
from lvdm.modules.networks.openaimodel3d import (
    UNetModel, ResBlock, Downsample, Upsample,
    SpatialTransformer, TemporalTransformer
)
from lvdm.models.utils_diffusion import timestep_embedding

def extract_unet_layers(model):
    """
    UNet 모델에서 각 레이어를 추출하고, 레이어의 위치와 타입을 포함하여 딕셔너리에 저장합니다.
    블록 인덱스는 Downsample이나 Upsample 레이어 이후에만 증가합니다.
    """
    extracted_layers = OrderedDict()
    
    # 입력 블록 추출
    extracted_layers['input_blocks'] = []
    current_block_layers = []
    block_idx = 0
    for block in model.model.diffusion_model.input_blocks:
        for layer_idx, layer in enumerate(block):
            layer_info = {
                'layer': layer,
                'block_type': 'input',
                'block_idx': block_idx,
                'layer_idx': layer_idx,
                'layer_type': type(layer).__name__
            }
            current_block_layers.append(layer_info)
            if isinstance(layer, Downsample):
                # 현재 블록을 저장하고 블록 인덱스 증가
                extracted_layers['input_blocks'].append(current_block_layers)
                block_idx += 1
                current_block_layers = []
    # 마지막 블록 추가
    if current_block_layers:
        extracted_layers['input_blocks'].append(current_block_layers)
    
    # 미들 블록 추출
    extracted_layers['middle_block'] = []
    for layer_idx, layer in enumerate(model.model.diffusion_model.middle_block):
        layer_info = {
            'layer': layer,
            'block_type': 'middle',
            'layer_idx': layer_idx,
            'layer_type': type(layer).__name__
        }
        extracted_layers['middle_block'].append(layer_info)
    
    # 출력 블록 추출
    extracted_layers['output_blocks'] = []
    current_block_layers = []
    block_idx = 0
    for block in model.model.diffusion_model.output_blocks:
        for layer_idx, layer in enumerate(block):
            layer_info = {
                'layer': layer,
                'block_type': 'output',
                'block_idx': block_idx,
                'layer_idx': layer_idx,
                'layer_type': type(layer).__name__
            }
            current_block_layers.append(layer_info)
            if isinstance(layer, Upsample):
                # 현재 블록을 저장하고 블록 인덱스 증가
                extracted_layers['output_blocks'].append(current_block_layers)
                block_idx += 1
                current_block_layers = []
    # 마지막 블록 추가
    if current_block_layers:
        extracted_layers['output_blocks'].append(current_block_layers)
    
    # 최종 출력 레이어 추출
    extracted_layers['out'] = []
    for layer_idx, layer in enumerate(model.model.diffusion_model.out):
        layer_info = {
            'layer': layer,
            'block_type': 'out',
            'layer_idx': layer_idx,
            'layer_type': type(layer).__name__
        }
        extracted_layers['out'].append(layer_info)
    
    return extracted_layers

def apply_transformer_layer(h, layer_info, context, b):
    """
    Transformer 레이어를 적용하는 함수로, SpatialTransformer와 TemporalTransformer를 처리합니다.
    """
    layer = layer_info['layer']
    if isinstance(layer, SpatialTransformer):
        h = layer(h, context)
    elif isinstance(layer, TemporalTransformer):
        h = rearrange(h, '(b t) c h w -> b c t h w', b=b)
        h = layer(h, context)
        h = rearrange(h, 'b c t h w -> (b t) c h w')
    else:
        raise TypeError(f"Unexpected transformer layer type: {type(layer)}")
    return h

def forward_with_extracted_layers(model, x, timesteps, context=None, fps=16):
    """
    추출된 레이어를 사용하여 모델의 순전파 과정을 수행합니다.
    각 블록을 지날 때 해당 블록의 이름을 출력합니다.
    """
    # 레이어 추출
    extracted_layers = extract_unet_layers(model)
    
    # 시간 임베딩 생성
    t_emb = timestep_embedding(timesteps, model.model.diffusion_model.model_channels)
    emb = model.model.diffusion_model.time_embed(t_emb)
    
    # FPS 임베딩 처리
    if model.model.diffusion_model.fps_cond:
        if isinstance(fps, int):
            fps = torch.full_like(timesteps, fps)
        fps_emb = timestep_embedding(fps, model.model.diffusion_model.model_channels)
        emb += model.model.diffusion_model.fps_embedding(fps_emb)
    
    b, c, t, h, w = x.shape
    
    # context와 emb를 (b*t) 크기에 맞게 확장
    if context is not None:
        context = context.repeat_interleave(repeats=t, dim=0)
    emb = emb.repeat_interleave(repeats=t, dim=0)
    
    # 입력 텐서를 (B*T, C, H, W) 형태로 변환
    x = rearrange(x, 'b c t h w -> (b t) c h w')
    
    h = x.type(model.model.diffusion_model.dtype)
    hs = []
    
    # 입력 블록 처리
    for block_idx, block_layers in enumerate(extracted_layers['input_blocks']):
        print(f"Processing input_block {block_idx}")
        for layer_info in block_layers:
            layer = layer_info['layer']
            print(f"  Layer: {layer_info['layer_type']}")
            if isinstance(layer, ResBlock):
                h = layer(h, emb, batch_size=b)
            elif isinstance(layer, Downsample):
                h = layer(h)
            elif isinstance(layer, (SpatialTransformer, TemporalTransformer)):
                h = apply_transformer_layer(h, layer_info, context, b)
            else:
                h = layer(h)
        hs.append(h)

    # hs를 역순으로 뒤집음
    hs = hs[::-1]
    
    # 미들 블록 처리
    print("Processing middle_block")
    for layer_info in extracted_layers['middle_block']:
        layer = layer_info['layer']
        print(f"  Layer: {layer_info['layer_type']}")
        if isinstance(layer, ResBlock):
            h = layer(h, emb, batch_size=b)
        elif isinstance(layer, (SpatialTransformer, TemporalTransformer)):
            h = apply_transformer_layer(h, layer_info, context, b)
        else:
            h = layer(h)
    
    # 출력 블록 처리
    for block_idx, block_layers in enumerate(extracted_layers['output_blocks']):
        print(f"Processing output_block {block_idx}")
        h_prev = hs[block_idx]
        print(f"  h shape: {h.shape}, h_prev shape: {h_prev.shape}")
        h = torch.cat([h, h_prev], dim=1)
        print(f"  concatenated h shape: {h.shape}")
        for layer_info in block_layers:
            layer = layer_info['layer']
            print(f"  Layer: {layer_info['layer_type']}")
            if isinstance(layer, ResBlock):
                h = layer(h, emb, batch_size=b)
    
    # 최종 출력 레이어 처리
    print("Processing out_block")
    for layer_info in extracted_layers['out']:
        layer = layer_info['layer']
        print(f"  Layer: {layer_info['layer_type']}")
        h = layer(h)
    
    # 텐서 형태 복원: (B*T, C, H, W) -> (B, C, T, H, W)
    y = rearrange(h, '(b t) c h w -> b c t h w', b=b)
    return y, extracted_layers

# 사용 예시
# 모델 로드 및 초기화
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

config_path = 'configs/inference_t2v_512_v2.0.yaml'  # 실제 경로로 수정하세요
ckpt_path = 'scripts/evaluation/model.ckpt'  # 실제 경로로 수정하세요

config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
model = model.eval()

# 입력 데이터 생성
x = torch.randn(1, 4, 16, 64, 40)  # (B, C, T, H, W)
timesteps = torch.tensor([10])
context = torch.randn(1, 77, 1024)  # 예시로 텍스트 임베딩 사용

print(model.di)


# 함수 실행
output, extracted_layers = forward_with_extracted_layers(model, x, timesteps, context=context, fps=16)

# 출력 확인
print(output.shape)  # 예상 출력: torch.Size([1, 4, 16, 64, 40])

# 특정 위치의 레이어에 접근 예시
# 예를 들어, 첫 번째 입력 블록의 두 번째 레이어를 가져오고 싶다면:
first_input_block_second_layer = extracted_layers['input_blocks'][0][1]['layer']
print(f"First input block's second layer type: {type(first_input_block_second_layer).__name__}")