import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig

from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from datetime import timedelta

from dataset_spec import AudioTextDataset

import os
import json
import random
import torch
import torch.nn.functional as F

from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoTokenizer

from torch.utils.tensorboard import SummaryWriter  # <--- TensorBoard SummaryWriter 추가


# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from mmg_inference.auffusion_pipe_functions_copy_0123 import (
    encode_audio_prompt,
    ConditionAdapter,
    import_model_class_from_model_name_or_path,
    retrieve_latents
)


from MMG_audio_teacher_inference import run_inference
from run_audio_eval import evaluate_audio_metrics



def evaluate_model(accelerator, unet_model, vae, image_processor, text_encoder_list, adapter_list, tokenizer_list, csv_path, inference_path, inference_batch_size, pretrained_model_name_or_path, seed, duration, guidance_scale, num_inference_steps, eta_audio, eval_id, target_folder):
    """
    FAD, CLAP 등 계산을 위한 평가 함수.
    """

    unet_model.eval()

    inference_path = f"{inference_path}/{eval_id}"
    
    with torch.no_grad():
        # Inference
        # if eval_id != 'step_1':
        # run_inference(
        #     accelerator=accelerator,
        #     unet_model=unet_model,
        #     vae=vae,
        #     image_processor=image_processor,
        #     text_encoder_list=text_encoder_list,
        #     adapter_list=adapter_list,
        #     tokenizer_list=tokenizer_list,
        #     prompt_file=csv_path,
        #     savedir=inference_path,
        #     bs=inference_batch_size,
        #     pretrained_model_name_or_path=pretrained_model_name_or_path,
        #     seed=seed,
        #     duration=duration,
        #     guidance_scale=guidance_scale,
        #     num_inference_steps=num_inference_steps,
        #     eta_audio=eta_audio
        # )

        accelerator.wait_for_everyone()

        # TODO: real FAD, CLAP calculation
        fad, clap_avg, fvd, clip_avg= 0.0, 0.0, 0.0, 0.0

        # if accelerator.is_main_process:
        #     fad, clap_avg, _ = evaluate_audio_metrics(
        #         preds_folder=inference_path,
        #         target_folder=target_folder,
        #         metrics=['FAD','CLAP'],
        #         clap_model=1,
        #         device=accelerator.device
        #     )
        accelerator.wait_for_everyone()
        unet_model.train()


        return fad, clap_avg




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
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="이전 체크포인트 경로 (resume 시 사용, 없으면 새로 학습)")


    # evaluation 관련
    parser.add_argument("--eval_every", type=int, default=2, help="N 에폭마다 평가")
    parser.add_argument("--inference_batch_size", type=int, default=1, help="inference batch size")
    parser.add_argument("--inference_save_path", type=str, default="audio_teacher_lora", help="inference 저장 위치")
    parser.add_argument("--eta_audio", type=float, default=0.0, help="inference eta_audio")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="inference cfg guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--target_folder", type=str, default="target_folder", help="Path to the folder with GT audio files.")
    parser.add_argument("--vgg_csv_path", type=str, default="/home/jupyter/MMG_TA_dataset_audiocaps_wavcaps/vggsound_sparse_curated_292.csv", help="Path to the folder with GT audio files.")
    parser.add_argument("--vgg_target_folder", type=str, default="/home/jupyter/MMG_TA_dataset_audiocaps_wavcaps/vggsound_sparse_test_curated_final/audio", help="Path to the folder with GT audio files.")
    parser.add_argument("--vgg_inference_path", type=str, default="/home/jupyter/audio_lora_vggsound_sparse_inference", help="inference 저장 위치")
    parser.add_argument("--ac_csv_path", type=str, default="/home/jupyter/MMG_TA_dataset_audiocaps_wavcaps/vggsound_sparse_curated_292.csv", help="Path to the folder with GT audio files.")
    parser.add_argument("--ac_target_folder", type=str, default="/home/jupyter/MMG_TA_dataset_audiocaps_wavcaps/vggsound_sparse_test_curated_final/audio", help="Path to the folder with GT audio files.")
    parser.add_argument("--ac_inference_path", type=str, default="/home/jupyter/audio_lora_vggsound_sparse_inference", help="inference 저장 위치")
                                


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


    # eval로 인한 종료 지연 코드
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
        os.environ["WANDB_MODE"] = "offline"
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

    args.audio_model_name = args.pretrained_model_name_or_path

    # Audio Models
    audio_unet = UNet2DConditionModel.from_pretrained(args.audio_model_name, subfolder="unet")
    audio_unet.eval()
    for param in audio_unet.parameters():
        param.requires_grad = True

    # # LoRA config
    # lora_config = LoraConfig(
    #     r=128,
    #     lora_alpha=128,
    #     init_lora_weights=True,
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )
    # audio_unet.add_adapter(lora_config)


    # audio_unet = load_accelerator_ckpt(audio_unet, args.audio_lora_ckpt_path)

    if not os.path.isdir(args.audio_model_name):
        pretrained_model_name_or_path = snapshot_download(args.audio_model_name)
    else:
        pretrained_model_name_or_path = args.audio_model_name

    # 2-2) VAE 로드
    with torch.no_grad():
        audio_vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae"
        )
    audio_vae = audio_vae.to(device=device,dtype=dtype)
    audio_vae.requires_grad_(False)

    # 2-3) VAE scale factor 기반 ImageProcessor
    audio_vae_scale_factor = 2 ** (len(audio_vae.config.block_out_channels) - 1)
    audio_image_processor = VaeImageProcessor(vae_scale_factor=audio_vae_scale_factor)

    # 2-4) condition_config.json 기반으로 text_encoder_list, tokenizer_list, adapter_list 로딩
    audio_condition_json_path = os.path.join(pretrained_model_name_or_path, "condition_config.json")
    with open(audio_condition_json_path, "r", encoding="utf-8") as f:
        audio_condition_json_list = json.load(f)

    audio_text_encoder_list = []
    audio_tokenizer_list = []
    audio_adapter_list = []

    with torch.no_grad():
        for cond_item in audio_condition_json_list:
            # text encoder / tokenizer
            audio_text_encoder_path = os.path.join(pretrained_model_name_or_path, cond_item["text_encoder_name"])
            audio_tokenizer = AutoTokenizer.from_pretrained(audio_text_encoder_path)
            
            audio_text_encoder_cls = import_model_class_from_model_name_or_path(audio_text_encoder_path)
            audio_text_encoder = audio_text_encoder_cls.from_pretrained(audio_text_encoder_path)

            audio_text_encoder.requires_grad_(False)
            audio_text_encoder = audio_text_encoder.to(device=device, dtype=dtype)

            audio_tokenizer_list.append(audio_tokenizer)
            audio_text_encoder_list.append(audio_text_encoder)

            # condition adapter
            audio_adapter_path = os.path.join(pretrained_model_name_or_path, cond_item["condition_adapter_name"])
            audio_adapter = ConditionAdapter.from_pretrained(audio_adapter_path)
            audio_adapter.requires_grad_(False)
            audio_adapter = audio_adapter.to(device=device,dtype=dtype)

            audio_adapter_list.append(audio_adapter)


    seed = args.seed
    # PyTorch Generator 설정
    generator = torch.Generator(device=device).manual_seed(seed)
    random.seed(seed)

    unet_model = audio_unet

    # All params will be trained
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

    # Prepare
    unet_model, optimizer, train_loader, vae, image_processor, text_encoder_list, adapter_list = accelerator.prepare(
        unet_model, optimizer, train_loader, audio_vae, audio_image_processor, audio_text_encoder_list, audio_adapter_list
    )
    tokenizer_list = audio_tokenizer_list

    # ====== 체크포인트에서 resume하는 부분 추가 ======
    start_epoch = 0
    global_step = 0
    resume_batch_idx = 0
    # if args.resume_checkpoint is not None:
    #     # accelerator가 저장했던 전체 상태를 로드합니다.
    #     accelerator.load_state(args.resume_checkpoint)
    #     training_state_path = os.path.join(args.resume_checkpoint, "training_state.json")
    #     if os.path.exists(training_state_path):
    #         with open(training_state_path, "r") as f:
    #             training_state = json.load(f)
    #         global_step = training_state.get("global_step", 0)
    #         # 마지막 저장된 에폭 이후부터 재개하도록 (+1)
    #         start_epoch = training_state.get("epoch", 0) + 1
    #         resume_batch_idx = training_state.get("batch_idx", -1) + 1
    #         print(f"체크포인트로부터 학습 재개: 에폭 {start_epoch}부터, 글로벌 스텝 {global_step}")
    #     else:
    #         print("체크포인트 내 training_state.json 파일을 찾을 수 없어, 새롭게 시작합니다.")
    # # =================================================

    unet_model.train()
    currunt_idx = 0



    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir="/home/work/kby_hgh/MMG_01/tensorboard_audio_full_0603")


    for epoch in range(args.num_epochs):
        losses = []
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", disable=not accelerator.is_main_process)


        # # 초기 평가
        # if global_step==0:
        #     accelerator.wait_for_everyone()
        #     vgg_fad, vgg_clap_avg = evaluate_model(
        #         accelerator=accelerator,
        #         unet_model=unet_model,
        #         vae=vae,
        #         image_processor=image_processor,
        #         text_encoder_list=text_encoder_list,
        #         adapter_list=adapter_list,
        #         tokenizer_list=tokenizer_list,
        #         csv_path=args.vgg_csv_path,
        #         inference_path=args.vgg_inference_path,
        #         inference_batch_size=args.inference_batch_size,
        #         pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        #         seed=args.seed,
        #         duration=args.slice_duration,
        #         guidance_scale=args.guidance_scale,
        #         num_inference_steps=args.num_inference_steps,
        #         eta_audio=args.eta_audio,
        #         eval_id=f"step_{global_step}",  # 스텝 단위 표시
        #         target_folder=args.vgg_target_folder
        #     )
                
        #     if accelerator.is_main_process and writer is not None:
        #         writer.add_scalar("eval/vgg_fad", vgg_fad, global_step)
        #         writer.add_scalar("eval/vgg_clap_avg", vgg_clap_avg, global_step)

        #     accelerator.wait_for_everyone()

        #     # AC 데이터셋으로 평가
        #     ac_fad, ac_clap_avg = evaluate_model(
        #         accelerator=accelerator,
        #         unet_model=unet_model,
        #         vae=vae,
        #         image_processor=image_processor,
        #         text_encoder_list=text_encoder_list,
        #         adapter_list=adapter_list,
        #         tokenizer_list=tokenizer_list,
        #         csv_path=args.ac_csv_path,
        #         inference_path=args.ac_inference_path,
        #         inference_batch_size=args.inference_batch_size,
        #         pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        #         seed=args.seed,
        #         duration=args.slice_duration,
        #         guidance_scale=args.guidance_scale,
        #         num_inference_steps=args.num_inference_steps,
        #         eta_audio=args.eta_audio,
        #         eval_id=f"step_{global_step}",  # 스텝 단위 표시
        #         target_folder=args.ac_target_folder
        #     )
        #     accelerator.wait_for_everyone()

        #     # wandb 로깅
        #     if accelerator.is_main_process:
        #         wandb.log({
        #             "eval/ac_fad": ac_fad,
        #             "eval/ac_clap_avg": ac_clap_avg,
        #             "step": global_step
        #         })
        #     if accelerator.is_main_process and writer is not None:
        #         writer.add_scalar("eval/ac_fad", ac_fad, global_step)
        #         writer.add_scalar("eval/ac_clap_avg", ac_clap_avg, global_step)



        for batch_idx, batch in enumerate(loop):
            currunt_idx += 1
            if currunt_idx < global_step: # and epoch+1 <= start_epoch: 
                continue
           
            with accelerator.accumulate(unet_model):
                optimizer.zero_grad()
                
                spec = batch["spec"] # it is a spectrogram
                caption = batch["caption"]


                # # spec의 min, max, mean 출력
                # spec_min, spec_max, spec_mean = spec.min(), spec.max(), spec.mean()
                # print(f"Spec - Min: {spec_min:.6f}, Max: {spec_max:.6f}, Mean: {spec_mean:.6f}")


                caption_text = caption
                #spectrograms = (spec + 1) / 2 

                # spectrogram의 min, max, mean 출력
                #spectrograms_min, spectrograms_max, spectrograms_mean = spectrograms.min(), spectrograms.max(), spectrograms.mean()
                #print(f"Spectrogram - Min: {spectrograms_min:.6f}, Max: {spectrograms_max:.6f}, Mean: {spectrograms_mean:.6f}")

                image = image_processor.preprocess(spec)  # 대략 [1, C, H, W] 형태 반환 가정

                # # image의 min, max, mean 출력
                # image_min, image_max, image_mean = image.min(), image.max(), image.mean()
                # print(f"Image - Min: {image_min:.6f}, Max: {image_max:.6f}, Mean: {image_mean:.6f}")

                # 텍스트/오디오 프롬프트 인코딩
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



                bsz = audio_latent.size(0)

                # 10% null conditioning: 배치 내 각 샘플 중 10%를 무작위로 선택하여 조건 정보 제거
                audio_null_text_emb = torch.zeros_like(audio_text_embed)  # [bsz, 77, 768]
                mask = (torch.rand(bsz, 1, 1, device=device) < 0.1)    # [bsz, 1, 1]
                text_emb = torch.where(mask, audio_null_text_emb, audio_text_embed)  # [bsz, 77, 768]

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

                # if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                global_step += 1

                if accelerator.is_main_process:
                    avg_loss = sum(losses) / len(losses)

                    # 텐서보드에 로그
                    if writer is not None:
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("epoch", epoch, global_step)

                    wandb.log({
                        "train/loss": avg_loss,
                        "epoch": epoch + 1,
                        "step": global_step
                    })
                    losses = []

                # -----  스텝 단위로 평가 -----
                if global_step % args.eval_every == 0: # global_step==1 or
                    accelerator.wait_for_everyone()

                    # VGG 데이터셋으로 평가
                    vgg_fad, vgg_clap_avg = evaluate_model(
                        accelerator=accelerator,
                        unet_model=unet_model,
                        vae=vae,
                        image_processor=image_processor,
                        text_encoder_list=text_encoder_list,
                        adapter_list=adapter_list,
                        tokenizer_list=tokenizer_list,
                        csv_path=args.vgg_csv_path,
                        inference_path=args.vgg_inference_path,
                        inference_batch_size=args.inference_batch_size,
                        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                        seed=args.seed,
                        duration=args.slice_duration,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        eta_audio=args.eta_audio,
                        eval_id=f"step_{global_step}",  # 스텝 단위 표시
                        target_folder=args.vgg_target_folder
                    )
                    accelerator.wait_for_everyone()


                    # wandb 로깅
                    if accelerator.is_main_process:
                        wandb.log({
                            "eval/vggsparse_fad": vgg_fad,
                            "eval/vggsparse_clap_avg": vgg_clap_avg,
                            "step": global_step
                        })
                    if accelerator.is_main_process and writer is not None:
                        writer.add_scalar("eval/vgg_fad", vgg_fad, global_step)
                        writer.add_scalar("eval/vgg_clap_avg", vgg_clap_avg, global_step)

                    # AC 데이터셋으로 평가
                    ac_fad, ac_clap_avg = evaluate_model(
                        accelerator=accelerator,
                        unet_model=unet_model,
                        vae=vae,
                        image_processor=image_processor,
                        text_encoder_list=text_encoder_list,
                        adapter_list=adapter_list,
                        tokenizer_list=tokenizer_list,
                        csv_path=args.ac_csv_path,
                        inference_path=args.ac_inference_path,
                        inference_batch_size=args.inference_batch_size,
                        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                        seed=args.seed,
                        duration=args.slice_duration,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        eta_audio=args.eta_audio,
                        eval_id=f"step_{global_step}",  # 스텝 단위 표시
                        target_folder=args.ac_target_folder
                    )
                    accelerator.wait_for_everyone()


                    # wandb 로깅
                    if accelerator.is_main_process:
                        wandb.log({
                            "eval/ac_fad": ac_fad,
                            "eval/ac_clap_avg": ac_clap_avg,
                            "step": global_step
                        })
                    if accelerator.is_main_process and writer is not None:
                        writer.add_scalar("eval/ac_fad", ac_fad, global_step)
                        writer.add_scalar("eval/ac_clap_avg", ac_clap_avg, global_step)


                # Save checkpoint
                # 체크포인트 저장 (평가 주기마다)
                if global_step > 0 and (global_step % args.eval_every == 0) and accelerator.is_main_process:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                    accelerator.save_state(ckpt_dir)
                    # 현재 학습 상태(에폭, 글로벌 스텝, 배치 인덱스)를 JSON 파일로 저장
                    training_state = {"global_step": global_step, "epoch": epoch, "batch_idx": batch_idx}
                    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                        json.dump(training_state, f)
                    print(f"[Epoch {epoch}] Checkpoint saved at: {ckpt_dir}")
                # ----------------------------------------------------

                if accelerator.is_main_process:
                    loop.set_postfix({"loss": loss.item()})

    if accelerator.is_main_process:
        wandb.finish()

    print("Training Complete.")



import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

