import torch
from diffusers import PNDMScheduler, UNet2DConditionModel

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



def forward_with_extracted_layers_unet(
    audio_unet,
    audio_latents,
    audio_timestep,
    audio_encoder_hidden_states,
    audio_attention_mask=None,
    audio_cross_attention_kwargs=None
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
        audio_cross_attention_kwargs (Optional[Dict[str, Any]], optional): 크로스 어텐션에 전달할 추가 인자. 기본값은 None.
        audio_added_cond_kwargs (Optional[Dict[str, torch.Tensor]], optional): 추가 조건 인자. 기본값은 None.

    Returns:
        torch.Tensor 또는 Tuple[torch.Tensor]: UNet의 출력 텐서.
    """
    
    # 0. 입력 샘플 중심화
    if audio_unet.config.center_input_sample:
        print("Centering input sample")
        audio_latents = 2 * audio_latents - 1.0
    # 1. 타임 임베딩 처리
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
    # time_embed_act 적용
    if audio_unet.time_embed_act is not None:
        audio_emb = audio_unet.time_embed_act(audio_emb)
    # 1.7. 인코더 숨겨진 상태 처리
    print("Processing encoder hidden states")
    if audio_unet.encoder_hid_proj is not None:
        audio_encoder_hidden_states = audio_unet.encoder_hid_proj(audio_encoder_hidden_states)
    # 2. pre-process (conv_in)
    print("Processing conv_in")
    audio_hidden_states = audio_unet.conv_in(audio_latents)
    audio_down_block_res_samples = (audio_hidden_states,)
    


    

    # 3. down_blocks 처리
    print("Processing down_blocks")
    print(f"  [down_block] 0")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(down_block= audio_unet.down_blocks[0], hidden_states = audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, output_states=audio_down_block_res_samples)
    print(f"    [down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[0].downsamplers[0](audio_hidden_states)
    audio_down_block_res_samples = audio_down_block_res_samples + (audio_hidden_states,)   
    print(f"  [down_block] 1")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(down_block= audio_unet.down_blocks[1], hidden_states = audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, output_states=audio_down_block_res_samples)
    print(f"    [down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[1].downsamplers[0](audio_hidden_states)
    audio_down_block_res_samples = audio_down_block_res_samples + (audio_hidden_states,)   
    print(f"  [down_block] 2")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(down_block= audio_unet.down_blocks[2], hidden_states = audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, output_states=audio_down_block_res_samples)
    print(f"    [down_block][downsampler]")
    audio_hidden_states = audio_unet.down_blocks[2].downsamplers[0](audio_hidden_states)
    audio_down_block_res_samples = audio_down_block_res_samples + (audio_hidden_states,)   
    print(f"  [down_block] 3")
    audio_hidden_states, audio_down_block_res_samples = audio_down_blocks(down_block= audio_unet.down_blocks[3], hidden_states = audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, output_states=audio_down_block_res_samples)

    # 4. mid_block 처리
    print("Processing mid_block")
    audio_hidden_states = audio_mid_blocks(audio_unet, audio_hidden_states, audio_emb, audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs)

    # 5. up_blocks 처리
    print("Processing up_blocks")
    print(f"  [up_block] 0")
    if len(audio_down_block_res_samples) == 0:
        raise ValueError(f"No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[0].resnets) :]
    audio_down_block_res_samples = audio_down_block_res_samples[: -len(audio_unet.up_blocks[0].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(audio_unet.up_blocks[0], hidden_states=audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, res_hidden_states_tuple=audio_res_hidden_states_tuple)
    print(f"    [up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[0].upsamplers[0](audio_hidden_states)
    print(f"  [up_block] 1")
    if len(audio_down_block_res_samples) == 0:
        raise ValueError(f"No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[1].resnets) :]
    audio_down_block_res_samples = audio_down_block_res_samples[: -len(audio_unet.up_blocks[1].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(audio_unet.up_blocks[1], hidden_states=audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, res_hidden_states_tuple=audio_res_hidden_states_tuple)
    print(f"    [up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[1].upsamplers[0](audio_hidden_states)
    print(f"  [up_block] 2")
    if len(audio_down_block_res_samples) == 0:
        raise ValueError(f"No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[2].resnets) :]
    audio_down_block_res_samples = audio_down_block_res_samples[: -len(audio_unet.up_blocks[2].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(audio_unet.up_blocks[2], hidden_states=audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, res_hidden_states_tuple=audio_res_hidden_states_tuple)
    print(f"    [up_block][upsampler]")
    audio_hidden_states = audio_unet.up_blocks[2].upsamplers[0](audio_hidden_states)
    print(f"  [up_block] 3")
    if len(audio_down_block_res_samples) == 0:
        raise ValueError(f"No residual samples available for up_block")
    audio_res_samples = audio_down_block_res_samples[-len(audio_unet.up_blocks[3].resnets) :]
    audio_down_block_res_samples = audio_down_block_res_samples[: -len(audio_unet.up_blocks[3].resnets)]
    audio_res_hidden_states_tuple = audio_res_samples 
    audio_hidden_states = audio_up_blocks(audio_unet.up_blocks[3], hidden_states=audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states, emb=audio_emb, attention_mask=audio_attention_mask, cross_attention_kwargs=audio_cross_attention_kwargs, res_hidden_states_tuple=audio_res_hidden_states_tuple)

    # 6. 후처리 (conv_norm_out 및 conv_out)
    if audio_unet.conv_norm_out is not None:
        print("Processing conv_norm_out")
        audio_hidden_states = audio_unet.conv_norm_out(audio_hidden_states)
        if audio_unet.conv_act is not None:
            print("Processing conv_act")
            audio_hidden_states = audio_unet.conv_act(audio_hidden_states)
    print("Processing conv_out")
    audio_hidden_states = audio_unet.conv_out(audio_hidden_states)

    return audio_hidden_states


def main():
    """
    메인 함수: 모델을 로드하고, 입력을 준비한 후,
    원본 모델과 레이어 추출 함수를 통해 출력을 비교합니다.
    """
    # 디바이스 및 데이터 타입 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    batch_size = 1
    generator = torch.Generator(device=device).manual_seed(42)

    # UNet 모델 로드
    print("Loading UNet2DConditionModel...")
    audio_unet = UNet2DConditionModel.from_pretrained("auffusion/auffusion-full", subfolder="unet").to(device, dtype)
    audio_unet.eval()
    # 입력 데이터 준비
    audio_height = 64  # 필요한 값으로 변경 가능
    audio_width = 64  # 필요한 값으로 변경 가능 

    # 텍스트 인코딩 (예시로 무작위 텐서 사용)
    audio_encoder_hidden_states = torch.randn((batch_size, 77, audio_unet.config.cross_attention_dim), device=device, dtype=dtype)

    # 노이즈 생성
    audio_latents = torch.randn(
        (batch_size, audio_unet.config.in_channels, audio_height // 8, audio_width // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # 타임스텝 설정
    print("Setting up scheduler...")
    audio_scheduler = PNDMScheduler.from_pretrained("auffusion/auffusion", subfolder="scheduler")
    audio_num_inference_steps = 50
    audio_scheduler.set_timesteps(audio_num_inference_steps, device=device)
    audio_timesteps = audio_scheduler.timesteps
    audio_t = audio_timesteps[0]

    # 원본 UNet 출력 계산
    print("Computing original UNet output...")
    with torch.no_grad():
        audio_original_output = audio_unet(
            audio_latents,
            audio_t,
            encoder_hidden_states=audio_encoder_hidden_states
        )[0]

    # 수정된 함수로 출력 계산
    print("Computing extracted layers UNet output...")
    with torch.no_grad():
        audio_extract_model_output = forward_with_extracted_layers_unet(
            audio_unet=audio_unet,
            audio_latents=audio_latents,
            audio_timestep=audio_t,
            audio_encoder_hidden_states=audio_encoder_hidden_states
        )



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
    compare_outputs(audio_original_output, audio_extract_model_output)


if __name__ == "__main__":
    main()
