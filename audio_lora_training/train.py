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

# 사용자 정의 모듈 임포트
from dataset import AudioTextDataset

from preprocess.auffusion_pipe_functions import (
    encode_audio_prompt, ConditionAdapter, import_model_class_from_model_name_or_path, retrieve_latents
)
from MMG_audio_teacher_inference import run_inference
from run_audio_eval import evaluate_audio_metrics

def evaluate_model(accelerator, unet_model, csv_path, inference_path, inference_batch_size, pretrained_model_name_or_path, seed, duration, guidance_scale, num_inference_steps, eta_audio, epoch, target_folder):
    """
    FAD, CLAP 등 계산을 위한 평가 함수.
    """

    unet_model.eval()

    inference_path = f"{inference_path}/{epoch}"
    
    with torch.no_grad():  
        # Inference 
        run_inference(
            accelerator=accelerator,
            unet_model=unet_model,
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

        # TODO: real FAD, CLAP calculation
        fad, clap = evaluate_audio_metrics(
            preds_folder=inference_path,
            target_folder=target_folder,
            metrics=[FAD,CLAP],
            clap_model=1,
            device=accelerator.device
        )

    unet_model.train()

    return fad, flap


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
    parser.add_argument("--guidance_scale", type=int, default=1, help="inference cfg guidance scale")
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

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    device = accelerator.device
    dtype = accelerator.dtype

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
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        sample_rate=args.sample_rate,
        slice_duration=args.slice_duration,
        hop_size=args.hop_size,
        n_mels=args.n_mels,
        seed=args.seed,
        device=device,
        dtype=dtype
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    # UNet + LoRA
    unet_model = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    ).to(device)
    unet_model.eval()
    for param in unet_model.parameters():
        param.requires_grad = False

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet_model.add_adapter(lora_config)

    # Only LoRA params will be trained
    trainable_params = [p for p in unet_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    # Prepare
    unet_model, optimizer, train_loader, val_loader = accelerator.prepare(
        unet_model, optimizer, train_loader, val_loader
    )

    global_step = 0
    unet_model.train()

    for epoch in range(args.num_epochs):
        losses = []
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", disable=not accelerator.is_main_process)

        for step, batch in enumerate(loop):
            audio_latent = batch["audio_latent"]
            text_emb = batch["text_emb"]

            bsz = audio_latent.size(0)

            # 10% null conditioning
            audio_null_text_emb = torch.zeros_like(text_emb)
            mask = (torch.rand(bsz, 1, device=device) < 0.1)
            text_emb = torch.where(mask, audio_null_text_emb, text_emb)

            # Sample random timestep
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()

            # Add noise
            noise = torch.randn_like(audio_latent)
            noised_latent = noise_scheduler.add_noise(original_samples=audio_latent, noise=noise, timesteps=timesteps)

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


        # Evaluate every n epochs
        if (epoch + 1) % args.eval_every == 0:
            if accelerator.is_main_process:
                fad, clap_avg, clap_std= evaluate_model(
                    accelerator=accelerator,
                    unet_model=unet_model,
                    csv_path=args.csv_path,
                    inference_path=args.inference_path,
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

                wandb.log({
                    "eval/fad": fad,
                    "eval/clap_avg": clap_avg,
                    "eval/clap_std": clap_std,
                    "epoch": epoch + 1,
                    "step": global_step
                })

        # Save checkpoint
        if (epoch + 1) % args.save_checkpoint == 0 and accelerator.is_main_process:
            # accelerator.save_state expects a directory, so we use a folder name
            ckpt_dir = os.path.join(args.output_dir, f"audio_teacher_unet_checkpoint/checkpoint-epoch-{epoch+1}")
            accelerator.save_state(ckpt_dir)
            print(f"[Epoch {epoch+1}] Checkpoint saved at: {ckpt_dir}")

    if accelerator.is_main_process:
        wandb.finish()

    print("Training Complete.")


if __name__ == "__main__":
    main()
