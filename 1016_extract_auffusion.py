import torch
from diffusers import PNDMScheduler, UNet2DConditionModel


def forward_with_extracted_layers_unet(
    unet,
    sample,
    timestep,
    encoder_hidden_states,
    class_labels=None,
    timestep_cond=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    added_cond_kwargs=None,
    return_dict=False,
):
    """
    UNet2DConditionModel의 forward 함수를 레이어 단위로 재현한 함수입니다.
    각 블록과 내부의 레이어를 순회하며 처리 단계를 출력합니다.

    Args:
        unet (UNet2DConditionModel): UNet2DConditionModel 인스턴스.
        sample (torch.Tensor): 노이즈가 추가된 입력 텐서, 형태는 (batch, channel, height, width).
        timestep (torch.Tensor, float, int): 노이즈를 제거할 타임스텝.
        encoder_hidden_states (torch.Tensor): 인코더의 숨겨진 상태, 형태는 (batch, sequence_length, feature_dim).
        class_labels (Optional[torch.Tensor], optional): 클래스 레이블. 기본값은 None.
        timestep_cond (Optional[torch.Tensor], optional): 타임스텝 조건. 기본값은 None.
        attention_mask (Optional[torch.Tensor], optional): 어텐션 마스크. 기본값은 None.
        cross_attention_kwargs (Optional[Dict[str, Any]], optional): 크로스 어텐션에 전달할 추가 인자. 기본값은 None.
        added_cond_kwargs (Optional[Dict[str, torch.Tensor]], optional): 추가 조건 인자. 기본값은 None.
        return_dict (bool, optional): 결과를 딕셔너리 형태로 반환할지 여부. 기본값은 False.

    Returns:
        torch.Tensor 또는 Tuple[torch.Tensor]: UNet의 출력 텐서.
    """
    # 초기화

    upsample_size = None

    # 0. 입력 샘플 중심화
    if unet.config.center_input_sample:
        print("Centering input sample")
        sample = 2 * sample - 1.0

    # 1. 타임 임베딩 처리
    print("Processing time embeddings")
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == 'mps'
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timestep], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = unet.time_proj(timesteps).to(dtype=sample.dtype)
    emb = unet.time_embedding(t_emb)

    # time_embed_act 적용
    if unet.time_embed_act is not None:
        emb = unet.time_embed_act(emb)

    # 1.7. 인코더 숨겨진 상태 처리
    print("Processing encoder hidden states")
    if unet.encoder_hid_proj is not None:
        encoder_hidden_states = unet.encoder_hid_proj(encoder_hidden_states)

    # 2. pre-process (conv_in)
    print("Processing conv_in")
    hidden_states = unet.conv_in(sample)



    down_block_res_samples = (hidden_states,)
    # 3. down_blocks 처리
    print("Processing down_blocks")
    for i, down_block in enumerate(unet.down_blocks):

        output_states = ()

        print(f"  [down_block {i}]")
        if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
            # CrossAttention 있는 경우
            for j, (resnet, attn) in enumerate(zip(down_block.resnets, down_block.attentions)):
                print(f"    [down_block {i}][resnet {j}]")
                hidden_states = resnet(hidden_states, emb)

                print(f"    [down_block {i}][attention {j}]")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                output_states = output_states + (hidden_states,)
        else:
            # CrossAttention 없는 경우
            for j, resnet in enumerate(down_block.resnets):
                print(f"    [down_block {i}][resnet {j}]")
                hidden_states = resnet(hidden_states, emb)
                output_states = output_states + (hidden_states,)

        # 다운샘플러 처리
        if down_block.downsamplers is not None and len(down_block.downsamplers) > 0:
            for k, downsampler in enumerate(down_block.downsamplers):
                print(f"    [down_block {i}][downsampler {k}]")
                hidden_states = downsampler(hidden_states)
                output_states = output_states + (hidden_states,)
        down_block_res_samples += output_states

        # Residual samples 저장 (final hidden_states of the block)
        #down_block_res_samples.append(hidden_states)

    # 4. mid_block 처리
    print("Processing mid_block")
    if unet.mid_block is not None:
        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            print(f"  [mid_block][resnet 0]")
            hidden_states = unet.mid_block.resnets[0](hidden_states, emb)
            for j, (resnet, attn) in enumerate(zip(unet.mid_block.resnets[1:], unet.mid_block.attentions)):

                print(f"  [mid_block][attention {j}]")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                
                print(f"  [mid_block][resnet {j+1}]")
                hidden_states = resnet(hidden_states, emb)
        else:
            for j, resnet in enumerate(unet.mid_block.resnets):
                print(f"  [mid_block][resnet {j}]")
                hidden_states = resnet(hidden_states, emb)

    # 5. up_blocks 처리
    print("Processing up_blocks")
    for i, up_block in enumerate(unet.up_blocks):
        print(f"  [up_block {i}]")
        is_final_block = i == len(unet.up_blocks) - 1
        # Residual sample 하나 가져오기
        if len(down_block_res_samples) == 0:
            raise ValueError(f"No residual samples available for up_block {i}")

        #residual = down_block_res_samples.pop()
        res_samples = down_block_res_samples[-len(up_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]

        res_hidden_states_tuple = res_samples 

        # Spatial dimensions 확인 및 조정 (왜함?)
        #if hidden_states.shape[-2:] != residual.shape[-2:]:
        #    print(f"    Adjusting spatial size in up_block {i}")
        #    hidden_states = torch.nn.functional.interpolate(hidden_states, size=residual.shape[-2:], mode='nearest')

        # resnets and attentions
        if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
            for j, (resnet, attn) in enumerate(zip(up_block.resnets, up_block.attentions)):
        
                # 채널 차원에서 concatenation
                #print(f"    [up_block {i}][concat {j}]")
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)        
                
                
                print(f"    [up_block {i}][resnet {j}]")
                hidden_states = resnet(hidden_states, emb)

                print(f"    [up_block {i}][attention {j}]")
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
        else:
            for j, resnet in enumerate(up_block.resnets):

                 # 채널 차원에서 concatenation
                #print(f"    [up_block {i}][concat {j}]")
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                #print("h", hidden_states.shape)
                #print("r", res_hidden_states.shape)
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)        

                print(f"    [up_block {i}][resnet {j}]")
                hidden_states = resnet(hidden_states, emb)

        # 업샘플러 처리
        if up_block.upsamplers is not None and len(up_block.upsamplers) > 0:
            for k, upsampler in enumerate(up_block.upsamplers):
                print(f"    [up_block {i}][upsampler {k}]")
                hidden_states = upsampler(hidden_states)

    # 6. 후처리 (conv_norm_out 및 conv_out)
    if unet.conv_norm_out is not None:
        print("Processing conv_norm_out")
        hidden_states = unet.conv_norm_out(hidden_states)
        if unet.conv_act is not None:
            print("Processing conv_act")
            hidden_states = unet.conv_act(hidden_states)
    print("Processing conv_out")
    hidden_states = unet.conv_out(hidden_states)

    if not return_dict:
        return (hidden_states,)
    else:
        return {"sample": hidden_states}


def main():
    """
    메인 함수: 모델을 로드하고, 입력을 준비한 후,
    원본 모델과 레이어 추출 함수를 통해 출력을 비교합니다.
    """
    # 디바이스 및 데이터 타입 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # UNet 모델 로드
    print("Loading UNet2DConditionModel...")
    unet = UNet2DConditionModel.from_pretrained("auffusion/auffusion", subfolder="unet").to(device, dtype)

    # 입력 데이터 준비
    batch_size = 1
    height = 256  # 필요한 값으로 변경 가능
    width = 1024  # 필요한 값으로 변경 가능

    generator = torch.Generator(device=device).manual_seed(42)

    # 텍스트 인코딩 (예시로 무작위 텐서 사용)
    condition_dim = unet.config.cross_attention_dim
    encoder_hidden_states = torch.randn((batch_size, 77, condition_dim), device=device, dtype=dtype)

    # 노이즈 생성
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # 타임스텝 설정
    print("Setting up scheduler...")
    scheduler = PNDMScheduler.from_pretrained("auffusion/auffusion", subfolder="scheduler")
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    t = timesteps[0]

    # 원본 UNet 출력 계산
    print("Computing original UNet output...")
    with torch.no_grad():
        original_output = unet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

    # 수정된 함수로 출력 계산
    print("Computing extracted layers UNet output...")
    with torch.no_grad():
        extract_model_output = forward_with_extracted_layers_unet(
            unet,
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

    # 출력 비교 함수
    def compare_outputs(output1: torch.Tensor, output2: torch.Tensor, tol: float = 1e-5):
        """
        두 출력 텐서를 비교합니다.

        Args:
            output1 (torch.Tensor): 첫 번째 출력 텐서.
            output2 (torch.Tensor): 두 번째 출력 텐서.
            tol (float, optional): 허용 오차. 기본값은 1e-5.
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
    print("Comparing outputs...")
    compare_outputs(original_output, extract_model_output)


if __name__ == "__main__":
    main()
