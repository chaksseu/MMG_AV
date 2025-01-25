import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig

from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from datetime import timedelta

#from dataset import AudioTextDataset
from dataset_spec import AudioTextDataset


from mmg_inference.auffusion_pipe_functions_copy_0123 import (
    encode_audio_prompt, ConditionAdapter, import_model_class_from_model_name_or_path, retrieve_latents
)

import os
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoTokenizer



import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))




from preprocess.converter_copy_0123 import (
    get_mel_spectrogram_from_audio,
    normalize_spectrogram,
)
from preprocess.utils import pad_spec

from mmg_inference.auffusion_pipe_functions_copy_0123 import (
    encode_audio_prompt,
    ConditionAdapter,
    import_model_class_from_model_name_or_path,
    retrieve_latents,
)


from MMG_audio_teacher_inference import run_inference
from run_audio_eval import evaluate_audio_metrics



def evaluate_model(accelerator, unet_model, vae, image_processor, text_encoder_list, adapter_list, tokenizer_list, csv_path, inference_path, inference_batch_size, pretrained_model_name_or_path, seed, duration, guidance_scale, num_inference_steps, eta_audio, epoch, target_folder):
    """
    FAD, CLAP 등 계산을 위한 평가 함수.
    """

    unet_model.eval()

    inference_path = f"{inference_path}/{epoch}"
    
    with torch.no_grad():
        # Inference
        if epoch != 1:
            run_inference(
                accelerator=accelerator,
                unet_model=unet_model,
                vae=vae,
                image_processor=image_processor,
                text_encoder_list=text_encoder_list,
                adapter_list=adapter_list,
                tokenizer_list=tokenizer_list,
                prompt_file=csv_path,
                savedir=inference_path,
                bs=inference_batch_size,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                seed=seed,
                duration=duration,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                eta_audio=eta_audio
            )

        accelerator.wait_for_everyone()

        # TODO: real FAD, CLAP calculation
        fad, clap_avg, clap_std = -1111, -1111, -1111
        if accelerator.is_main_process:
            fad, clap_avg, clap_std = evaluate_audio_metrics(
                preds_folder=inference_path,
                target_folder=target_folder,
                metrics=['FAD','CLAP'],
                clap_model=1,
                device=accelerator.device
            )
        unet_model.train()
        accelerator.wait_for_everyone()


        return fad, clap_avg, clap_std


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA on Audio+Text data")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV 파일 경로")
    parser.add_argument("--audio_dir", type=str, required=True, help="오디오(flac) 파일 폴더 경로")
    parser.add_argument("--train_batch_size", type=int, default=2, help="학습 배치 사이즈")
    parser.add_argument("--lr", type=float, default=1e-5, help="학습률")
    parser.add_argument("--num_epochs", type=int, default=10, help="에폭 수")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="그래디언트 누적 스텝")
    parser.add_argument("--wandb_project", type=str, default="audio_teacher_lora", help="WandB 프로젝트 이름")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="체크포인트 저장 디렉토리")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="혼합 정밀도 설정")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="auffusion/auffusion-full", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument("--save_checkpoint", type=int, default=100, help="save_checkpoint")

    # evaluation 관련
    parser.add_argument("--eval_every", type=int, default=2, help="N 에폭마다 평가")
    parser.add_argument("--inference_batch_size", type=int, default=1, help="inference batch size")
    parser.add_argument("--inference_save_path", type=str, default="audio_teacher_lora", help="inference 저장 위치")
    parser.add_argument("--eta_audio", type=float, default=0.0, help="inference eta_audio")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="inference cfg guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--target_folder", type=str, default="target_folder", help="Path to the folder with GT audio files.")


    # 기타 dataset 파라미터
    parser.add_argument("--sample_rate", type=int, default=16000, help="샘플링 레이트")
    parser.add_argument("--slice_duration", type=float, default=3.2, help="슬라이스 지속 시간 (초)")
    parser.add_argument("--hop_size", type=int, default=160, help="Hop size for mel spectrogram")
    parser.add_argument("--n_mels", type=int, default=256, help="Number of mel bands")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    # eval로 인한 멀
    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=3600)) 

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ipg_handler]
    )
    device = accelerator.device
    dtype = torch.float32

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    
    # wandb
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name="audio_lora_training")
    else:
        os.environ["WANDB_MODE"] = "offline"

    # Datasets
    train_dataset = AudioTextDataset(
        csv_path=args.csv_path,
        audio_dir=args.audio_dir,
        split="train",
        sample_rate=args.sample_rate,
        slice_duration=args.slice_duration,
        hop_size=args.hop_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    # UNet + LoRA
    unet_model = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    )
    unet_model.eval()
    for param in unet_model.parameters():
        param.requires_grad = False

    # LoRA config
    lora_config = LoraConfig(
        r=128,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet_model.add_adapter(lora_config)

    # Only LoRA params will be trained
    total_params = sum(p.numel() for p in unet_model.parameters())


    trainable_params = [p for p in unet_model.parameters() if p.requires_grad]
    total_trainable_params = sum(p.numel() for p in trainable_params)
    


    if accelerator.is_main_process:
        print(f"Total params: {total_params}")
        print(f"Total trainable parameters: {total_trainable_params}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )




    seed = args.seed
    # 랜덤 시드 설정
    generator = torch.Generator(device=device).manual_seed(seed)
    random.seed(seed)

    # ================================================
    # 2) 모델/토크나이저/어댑터 등 로딩 (auffusion 예시)
    # ================================================

    # 2-1) pretrained_model_name_or_path가 로컬 폴더가 아니면 snapshot_download
    if not os.path.isdir(args.pretrained_model_name_or_path):
        pretrained_model_name_or_path = snapshot_download(args.pretrained_model_name_or_path)
    else:
        pretrained_model_name_or_path = args.pretrained_model_name_or_path

    # 2-2) VAE 로드
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae"
        )
    vae.requires_grad_(False)

    # 2-3) VAE scale factor 기반 ImageProcessor
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # 2-4) condition_config.json 기반으로 text_encoder_list, tokenizer_list, adapter_list 로딩
    condition_json_path = os.path.join(pretrained_model_name_or_path, "condition_config.json")
    with open(condition_json_path, "r", encoding="utf-8") as f:
        condition_json_list = json.load(f)

    text_encoder_list = []
    tokenizer_list = []
    adapter_list = []

    with torch.no_grad():
        for cond_item in condition_json_list:
            # text encoder / tokenizer
            text_encoder_path = os.path.join(pretrained_model_name_or_path, cond_item["text_encoder_name"])
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            text_encoder_cls = import_model_class_from_model_name_or_path(text_encoder_path)
            text_encoder = text_encoder_cls.from_pretrained(text_encoder_path)

            text_encoder.requires_grad_(False)

            tokenizer_list.append(tokenizer)
            text_encoder_list.append(text_encoder)

            # condition adapter
            adapter_path = os.path.join(pretrained_model_name_or_path, cond_item["condition_adapter_name"])
            adapter = ConditionAdapter.from_pretrained(adapter_path)
            adapter.requires_grad_(False)
            adapter_list.append(adapter)
        



    # Prepare
    unet_model, optimizer, train_loader, vae, image_processor, text_encoder_list, adapter_list = accelerator.prepare(
        unet_model, optimizer, train_loader, vae, image_processor, text_encoder_list, adapter_list
    )


    global_step = 0
    unet_model.train()

    for epoch in range(args.num_epochs):
        losses = []
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", disable=not accelerator.is_main_process)


        # For test
        
        if (epoch + 1) % args.eval_every == 0:
            accelerator.wait_for_everyone()
            vgg_fad, vgg_clap_avg, vgg_clap_std= evaluate_model(
                accelerator=accelerator,
                unet_model=unet_model,
                vae=vae,
                image_processor=image_processor,
                text_encoder_list=text_encoder_list,
                adapter_list=adapter_list,
                tokenizer_list=tokenizer_list,
                csv_path="/home/rtrt5060/vggsound_sparse_curated_292.csv",
                inference_path="/home/rtrt5060/audio_lora_vggsound_sparse_inference",
                inference_batch_size=args.inference_batch_size,
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                seed=args.seed,
                duration=args.slice_duration,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                eta_audio=args.eta_audio,
                epoch=(epoch + 1),
                target_folder="/home/rtrt5060/vggsound_sparse_test_curated_final/audio"
                )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.log({
                    "eval/vggsparse_fad": vgg_fad,
                    "eval/vggsparse_clap_avg": vgg_clap_avg,
                    "eval/vggsparse_clap_std": vgg_clap_std
                })

            accelerator.wait_for_everyone()
            fad, clap_avg, clap_std= evaluate_model(
                accelerator=accelerator,
                unet_model=unet_model,
                vae=vae,
                image_processor=image_processor,
                text_encoder_list=text_encoder_list,
                adapter_list=adapter_list,
                tokenizer_list=tokenizer_list,
                csv_path=args.csv_path,
                inference_path=args.inference_save_path,
                inference_batch_size=args.inference_batch_size,
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                seed=args.seed,
                duration=args.slice_duration,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                eta_audio=args.eta_audio,
                epoch=(epoch + 1),
                target_folder=args.target_folder
                )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.log({
                    "eval/fad": fad,
                    "eval/clap_avg": clap_avg,
                    "eval/clap_std": clap_std,
                    "epoch": epoch + 1,
                    "step": global_step
                })
        

        for step, batch in enumerate(loop):
            audio_latent = batch["audio_latent"]
            caption = batch["caption"]

                    
            # ============================
            # (4) VAE, Adapter 등 사용
            # ============================
            # 예시 코드 상, spec을 [0,1] 범위 이미지처럼 가정하기 위해 (spec+1)/2 변환
            # 다만 실제로는 spec이 음수가 아닐 수도 있으니, 여기선 예시 그대로 두겠습니다.
            # spec: shape [n_mels, T]
            # 아래처럼 하면 (T, n_mels)별로 잘리는 점 주의 (원코드도 조금 애매합니다).
            # 일단 원 코드 흐름에 맞춰 진행:
            #print('####################')
            #print("spec.shape", spec.shape)
            spec = audio_latent
            caption_text = caption
            #spectrograms = [(row_ + 1) / 2 for row_ in spec]  # list of T개, 각각 shape [n_mels]
            spectrograms = (spec + 1) / 2 
            #print(111111111111111111111111111)
            #print("spec.dtype", spec.dtype)
            # image_processor 사용 예시
            # 실제 VaeImageProcessor가 어떻게 preprocess하는지는 diffusers 버전에 따라 다를 수 있습니다.
            # 보통 PIL Image나 [B,H,W,C] 텐서를 받는 식이 많으므로,
            # 아래 로직은 실제론 맞지 않을 수 있습니다. (개념적인 예시임)
            #print('####################')
            #print("spectrograms.shape", spectrograms.shape)
            #print(222222222222222222222222222)
            #print("spectrograms.dtype", spectrograms.dtype)
            image = image_processor.preprocess(spectrograms)  # 대략 [1, C, H, W] 형태 반환 가정
            #print(333333333333333333333333333)
            #print("image.dtype", image.dtype)
            # 텍스트/오디오 프롬프트 인코딩
            with accelerator.autocast():
                audio_text_embed = encode_audio_prompt(
                    text_encoder_list=text_encoder_list,
                    tokenizer_list=tokenizer_list,
                    adapter_list=adapter_list,
                    tokenizer_model_max_length=77,
                    dtype=dtype,
                    prompt=caption_text,
                    device=accelerator.device
                )

            # VAE 인코딩 -> latents
            with accelerator.autocast():
                vae_output = vae.encode(image)
                audio_latent = retrieve_latents(vae_output, generator=generator)
                audio_latent = vae.config.scaling_factor * audio_latent

            # 배치 차원 제거 (예: [1, ...] -> [...])
            #audio_latent = audio_latent.squeeze(0)
            #audio_text_embed = audio_text_embed.squeeze(0)
            text_emb = audio_text_embed





            bsz = audio_latent.size(0)

            # 10% null conditioning
            audio_null_text_emb = torch.zeros_like(text_emb)
            mask = (torch.rand(bsz, 77, 768, device=device) < 0.1)


            #print(f"mask shape: {mask.shape}")
            #print(f"audio_null_text_emb shape: {audio_null_text_emb.shape}")
            #print(f"text_emb shape: {text_emb.shape}")


            text_emb = torch.where(mask, audio_null_text_emb, text_emb)

            # Sample random timestep
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()

            # Add noise
            noise = torch.randn_like(audio_latent)
            noised_latent = noise_scheduler.add_noise(original_samples=audio_latent, noise=noise, timesteps=timesteps)



            #print(44444444444444444444444444444)
            #print("noised_latent.dtype", noised_latent.dtype)
            #print("text_emb.dtype", text_emb.dtype)
            #print("timesteps.dtype", timesteps.dtype)

            #print("shape(latents, text_embeds, timesteps)", noised_latent.shape, text_emb.shape, timesteps.shape)


            # Forward pass
            model_output = unet_model(
                noised_latent,
                timesteps,
                encoder_hidden_states=text_emb,
                return_dict=False,
            )[0]  # The first element is the predicted noise

            # MSE loss
            loss = F.mse_loss(model_output, noise)
            losses.append(loss.item())

            # Backprop
            accelerator.backward(loss)
            # Clip grad norm on unet_model parameters
            accelerator.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if accelerator.is_main_process:
                    avg_loss = sum(losses) / len(losses)
                    wandb.log({
                        "train/loss": avg_loss,
                        "epoch": epoch + 1,
                        "step": global_step
                    })
                    losses = []

            if accelerator.is_main_process:
                loop.set_postfix({"loss": loss.item()})


        # if (epoch + 1) % args.eval_every == 0:
        #     accelerator.wait_for_everyone()
        #     fad, clap_avg, clap_std= evaluate_model(
        #         accelerator=accelerator,
        #         unet_model=unet_model,
        #         vae=vae,
        #         image_processor=image_processor,
        #         text_encoder_list=text_encoder_list,
        #         adapter_list=adapter_list,
        #         tokenizer_list=tokenizer_list,
        #         csv_path="/home/rtrt5060/vggsound_sparse_curated_292.csv",
        #         inference_path="/home/rtrt5060/audio_lora_vggsound_sparse_inference",
        #         inference_batch_size=args.inference_batch_size,
        #         pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        #         seed=args.seed,
        #         duration=args.slice_duration,
        #         guidance_scale=args.guidance_scale,
        #         num_inference_steps=args.num_inference_steps,
        #         eta_audio=args.eta_audio,
        #         epoch=(epoch + 1),
        #         target_folder="/home/rtrt5060/vggsound_sparse_test_curated_final/audio"
        #         )
        #     if accelerator.is_main_process:
        #         wandb.log({
        #             "eval/vggsparse_fad": fad,
        #             "eval/vggsparse_clap_avg": clap_avg,
        #             "eval/vggsparse_clap_std": clap_std
        #         })

        #     accelerator.wait_for_everyone()
        #     fad, clap_avg, clap_std= evaluate_model(
        #         accelerator=accelerator,
        #         unet_model=unet_model,
        #         vae=vae,
        #         image_processor=image_processor,
        #         text_encoder_list=text_encoder_list,
        #         adapter_list=adapter_list,
        #         tokenizer_list=tokenizer_list,
        #         csv_path=args.csv_path,
        #         inference_path=args.inference_save_path,
        #         inference_batch_size=args.inference_batch_size,
        #         pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        #         seed=args.seed,
        #         duration=args.slice_duration,
        #         guidance_scale=args.guidance_scale,
        #         num_inference_steps=args.num_inference_steps,
        #         eta_audio=args.eta_audio,
        #         epoch=(epoch + 1),
        #         target_folder=args.target_folder
        #         )
        #     if accelerator.is_main_process:
        #         wandb.log({
        #             "eval/fad": fad,
        #             "eval/clap_avg": clap_avg,
        #             "eval/clap_std": clap_std,
        #             "epoch": epoch + 1,
        #             "step": global_step
        #         })



        # Save checkpoint
        if (epoch + 1) % args.save_checkpoint == 0 and accelerator.is_main_process:
            # accelerator.save_state expects a directory, so we use a folder name
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            accelerator.save_state(ckpt_dir)
            print(f"[Epoch {epoch+1}] Checkpoint saved at: {ckpt_dir}")

    if accelerator.is_main_process:
        wandb.finish()

    print("Training Complete.")



import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

