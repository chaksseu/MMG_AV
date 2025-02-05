import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer, CrossModalTransformer,
    Downsample, Upsample, TimestepBlock
)
from einops import rearrange
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta


from tqdm import tqdm
import wandb

from dataset import VideoTextDataset

from MMG_video_teacher_inference import run_inference
from run_video_eval import evaluate_video_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA on Audio+Text data")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV 파일 경로")
    parser.add_argument("--video_dir", type=str, required=True, help="비디오 파일 폴더 경로")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="체크포인트 저장 디렉토리")
    parser.add_argument("--wandb_project", type=str, default="audio_teacher_lora", help="WandB 프로젝트 이름")
    parser.add_argument("--train_batch_size", type=int, default=2, help="학습 배치 사이즈")
    parser.add_argument("--lr", type=float, default=1e-5, help="학습률")
    parser.add_argument("--num_epochs", type=int, default=10, help="에폭 수")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="그래디언트 누적 스텝")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="혼합 정밀도 설정")
    parser.add_argument("--eval_every", type=int, default=2, help="N 에폭마다 평가")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument("--save_checkpoint", type=int, default=100, help="save_checkpoint")
    parser.add_argument("--videocrafter_ckpt", type=str, default='scripts/evaluation/model.ckpt', help="Path to pretrained model.")
    parser.add_argument("--videocrafter_config", type=str, default='configs/inference_t2v_512_v2.0.yaml', help="Path to model config.")
    parser.add_argument("--video_fps", type=float, default=12.5, help="video_fps")
    parser.add_argument("--target_frames", type=int, default=40, help="num of frames")
    parser.add_argument("--inference_batch_size", type=int, default=1, help="inference batch size")
    parser.add_argument("--inference_save_path", type=str, default="audio_teacher_lora", help="inference 저장 위치")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="inference cfg guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--target_folder", type=str, default="target_folder", help="Path to the folder with GT files.")
    parser.add_argument("--height", type=int, default=320, help="height")
    parser.add_argument("--width", type=int, default=512, help="width")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim_eta")




    # vgg eval 관련 (필요시)
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--vgg_csv_path", type=str, default=None, help="CSV for vgg eval")
    parser.add_argument("--vgg_inference_save_path", type=str, default=None, help="Inference save path for vgg eval")
    parser.add_argument("--vgg_target_folder", type=str, default=None, help="Target folder for vgg eval")

    return parser.parse_args()


def evaluate_model(accelerator, unet_model, video_model, csv_path, inference_path,
                   inference_batch_size, seed, guidance_scale, height, width, frames,
                   ddim_eta, fps, num_inference_steps, step, target_folder):
                   
    unet_model.eval()
    inference_path = f"{inference_path}/step_{step}"
    
    with torch.no_grad():
        # if step != 0:
        run_inference(
            accelerator=accelerator,
            unet_model=unet_model,
            video_model=video_model,
            prompt_file=csv_path,
            savedir=inference_path,
            bs=inference_batch_size,
            seed=seed,
            unconditional_guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            frames=frames,
            ddim_eta=ddim_eta,
            fps=fps
        )

        accelerator.wait_for_everyone()

        # 실제 fvd, CLIP 등 계산
        fvd, clip_avg = -1111, -1111
        if accelerator.is_main_process:
            fvd, clip_avg = evaluate_video_metrics(
                preds_folder=inference_path,
                target_folder=target_folder,
                metrics=['fvd','clip'],
                device=accelerator.device,
                num_frames=frames
            )

        unet_model.train()
        accelerator.wait_for_everyone()

        return fvd, clip_avg


def main(args):
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    # eval로 인한 
    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=3600)) 

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ipg_handler]
    )
    device = accelerator.device
    # wandb
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name="video_lora_training")
    else:
        os.environ["WANDB_MODE"] = "offline"

    # Datasets
    train_dataset = VideoTextDataset(
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        split="train",
        target_frames=args.target_frames,
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    # Video UNet
    video_config = OmegaConf.load(args.videocrafter_config)
    video_model = instantiate_from_config(video_config.model)
    state_dict = torch.load(args.videocrafter_ckpt)['state_dict']
    video_model.load_state_dict(state_dict, strict=False)
    video_model.to(device=device).eval()
    video_unet = video_model.model.diffusion_model.eval()


    


    # 특정 부분만 학습
    for name, param in video_unet.named_parameters():
        if 'lora_block' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 전체 파라미터 개수 계산
    total_params = sum(p.numel() for p in video_unet.parameters())
    # 학습 가능한 파라미터 개수 계산
    trainable_params = sum(p.numel() for p in video_unet.parameters() if p.requires_grad)

    # 파라미터 개수 출력
    print(f"전체 파라미터 개수: {total_params}")
    print(f"학습 가능한 파라미터 개수: {trainable_params}")

    # 학습 가능한 파라미터 리스트 생성
    trainable_params_list = [p for p in video_unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=args.lr)

    # prepare
    video_unet, optimizer, train_loader = accelerator.prepare(
        video_unet, optimizer, train_loader
    )

    global_step = 0
    video_unet.train()

    for epoch in range(args.num_epochs):
        losses = []
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", disable=not accelerator.is_main_process)

        accelerator.wait_for_everyone()

        if args.vgg_csv_path is not None:
            vgg_fvd, vgg_clip_avg = evaluate_model(
                accelerator=accelerator,
                unet_model=video_unet,
                video_model=video_model,
                csv_path=args.vgg_csv_path,
                inference_path=args.vgg_inference_save_path,
                inference_batch_size=args.inference_batch_size,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                step=global_step,
                height=args.height,
                width=args.width,
                frames=args.target_frames,
                ddim_eta=args.ddim_eta,
                fps=args.video_fps,
                target_folder=args.vgg_target_folder
            )

            if accelerator.is_main_process:
                wandb.log({
                    "eval/vgg_fvd": vgg_fvd,
                    "eval/vgg_clip_avg": vgg_clip_avg,
                })

        fvd, clip_avg = evaluate_model(
                accelerator=accelerator,
                unet_model=video_unet,
                video_model=video_model,
                csv_path=args.csv_path,
                inference_path=args.inference_save_path,
                inference_batch_size=args.inference_batch_size,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                step=global_step,
                height=args.height,
                width=args.width,
                frames=args.target_frames,
                ddim_eta=args.ddim_eta,
                fps=args.video_fps,
                target_folder=args.target_folder
            )
        if accelerator.is_main_process:
            wandb.log({
                "eval/fvd": fvd,
                "eval/clip_avg": clip_avg,
            })


        for step, batch in enumerate(loop):
            with accelerator.accumulate(video_unet):
                video_tensor = batch["video_tensor"]  # [B, T, 3, 256, 256]
                caption = batch["caption"]

                batch_size = video_tensor.shape[0]

                # print('---------------------------------------------')
                # print("video_tensor", video_tensor.shape)
                # print("video_tensor", video_tensor.shape)
    
                video_tensor = video_tensor.permute(0, 4, 1, 2, 3)
                
                # print("video_tensor", video_tensor.shape)
                # print("video_tensor", video_tensor.shape)

                with torch.no_grad():
                    video_latent = video_model.encode_first_stage(video_tensor)
                    video_text_embed = video_model.get_learned_conditioning(caption)

                # Text dropout (optional)
                video_null_text_embed = torch.zeros_like(video_text_embed[:, :1, :])
                mask = (torch.rand(batch_size, 1, 1, device=video_text_embed.device) < 0.1)
                video_text_embed = torch.where(mask, video_null_text_embed, video_text_embed)

                timesteps = torch.randint(0, 1000, (batch_size,), device=device)

                noise_video = torch.randn_like(video_latent)
                noised_video_latent = video_model.q_sample(
                    x_start=video_latent, t=timesteps, noise=noise_video
                )


                fps_tensor = torch.full((batch_size,), args.video_fps, device=device)

                # UNet forward
                video_original_output = video_unet(
                    noised_video_latent,
                    timesteps,
                    context=video_text_embed,
                    fps=fps_tensor  # 필요하다면 float로
                )

                loss = F.mse_loss(video_original_output, noise_video)
                losses.append(loss.item())

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(video_unet.parameters(), max_norm=1.0)

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


                # -----  스텝 단위로 평가 -----
                if global_step % args.eval_every == 0: #global_step==1 or 
                    accelerator.wait_for_everyone()

                    if args.vgg_csv_path is not None:
                        vgg_fvd, vgg_clip_avg = evaluate_model(
                            accelerator=accelerator,
                            unet_model=video_unet,
                            video_model=video_model,
                            csv_path=args.vgg_csv_path,
                            inference_path=args.vgg_inference_save_path,
                            inference_batch_size=args.inference_batch_size,
                            seed=args.seed,
                            guidance_scale=args.guidance_scale,
                            num_inference_steps=args.num_inference_steps,
                            step=global_step,
                            height=args.height,
                            width=args.width,
                            frames=args.target_frames,
                            ddim_eta=args.ddim_eta,
                            fps=args.video_fps,
                            target_folder=args.vgg_target_folder
                        )

                        if accelerator.is_main_process:
                            wandb.log({
                                "eval/vgg_fvd": vgg_fvd,
                                "eval/vgg_clip_avg": vgg_clip_avg,
                            })

                    fvd, clip_avg = evaluate_model(
                            accelerator=accelerator,
                            unet_model=video_unet,
                            video_model=video_model,
                            csv_path=args.csv_path,
                            inference_path=args.inference_save_path,
                            inference_batch_size=args.inference_batch_size,
                            seed=args.seed,
                            guidance_scale=args.guidance_scale,
                            num_inference_steps=args.num_inference_steps,
                            step=global_step,
                            height=args.height,
                            width=args.width,
                            frames=args.target_frames,
                            ddim_eta=args.ddim_eta,
                            fps=args.video_fps,
                            target_folder=args.target_folder
                        )
                    if accelerator.is_main_process:
                        wandb.log({
                            "eval/fvd": fvd,
                            "eval/clip_avg": clip_avg,
                        })

                # Checkpoint 저장
                if global_step > 0 and (global_step % args.eval_every == 0) and accelerator.is_main_process:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                    accelerator.save_state(ckpt_dir)
                    print(f"[Epoch {global_step}] Checkpoint saved at: {ckpt_dir}")

                if accelerator.is_main_process:
                    loop.set_postfix({"loss": loss.item()})

    if accelerator.is_main_process:
        wandb.finish()

    print("Training Complete.")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    main(args)
