import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from omegaconf import OmegaConf
from functools import partial
import numpy as np

# 필요한 유틸들 로드 (VideoCrafter, lvdm 디렉토리 및 utils 모듈 경로 조정 필요)
from utils.utils import instantiate_from_config
from lvdm.models.ddpm3d import LatentDiffusion

##########################################################
# 예시용 VideoDataset (사용자 맞춤 구현 필요)
# (video: [C, T, H, W], prompt: string)
##########################################################
class VideoDataset(Dataset):
    def __init__(self, data_root, prompt_file, frames=16, image_size=(256,256)):
        # data_root 내 video 파일 목록
        self.video_files = sorted([os.path.join(data_root, fn) for fn in os.listdir(data_root) if fn.endswith('.mp4')])
        
        # prompt 파일 로딩
        with open(prompt_file, 'r') as f:
            self.prompts = [line.strip() for line in f if line.strip()]

        assert len(self.video_files) == len(self.prompts), "Video count != Prompt count"

        self.frames = frames
        self.image_size = image_size

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        prompt = self.prompts[idx]
        # 여기에 실제 비디오 로딩 로직 추가 필요
        # 예를 들어 decord, opencv 등을 이용해 영상을 읽고 [C,T,H,W] 형태로 resize & tensor 변환
        # 여기서는 dummy로 랜덤 tensor
        video = torch.randn(3, self.frames, self.image_size[0], self.image_size[1])
        return {"video": video, "prompt": prompt}


##########################################################
# 학습 함수
##########################################################
def train(
    config_path,
    ckpt_path,
    data_root,
    prompt_file,
    save_dir,
    max_steps=10000,
    batch_size=2,
    lr=1e-4,
    weight_decay=0.01,
    log_interval=100,
    save_interval=1000,
    mixed_precision='fp16' # 'no', 'fp16', 'bf16'
):

    # Accelerate 초기화
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    os.makedirs(save_dir, exist_ok=True)

    # Config 로드 및 모델 초기화
    config = OmegaConf.load(config_path)
    model_config = config.model
    model = instantiate_from_config(model_config)

    # Pretrained ckpt 로드 (vae, encoder 포함)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("Loaded pretrained ckpt:", ckpt_path)
        if missing: print("Missing keys:", missing)
        if unexpected: print("Unexpected keys:", unexpected)

    model.to(device)
    model.eval()

    # VAE, 인코더 등 freeze
    for param in model.first_stage_model.parameters():
        param.requires_grad = False
    if model.cond_stage_model is not None:
        for param in model.cond_stage_model.parameters():
            param.requires_grad = False

    # diffusion model (UNet)만 학습
    for param in model.model.diffusion_model.parameters():
        param.requires_grad = True

    # 데이터로더 구성
    dataset = VideoDataset(data_root, prompt_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Accelerate 준비
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    loss_fn = nn.MSELoss()

    global_step = 0
    model.train()

    for step, batch in enumerate(dataloader):
        if global_step >= max_steps:
            break

        video = batch["video"].to(device)   # [B,3,T,H,W]
        prompts = batch["prompt"]

        # Prompt -> 텍스트 임베딩
        with torch.no_grad():
            text_emb = model.get_learned_conditioning(list(prompts))
            cond = {"c_crossattn": [text_emb]} 
            # 필요하다면 fps embedding 등 추가 cond 가능 (cond["fps"] = fps_tensor)

            # 비디오를 latent로 인코딩
            z = model.encode_first_stage(video)  # [B, latentC, T, H/8, W/8]
            b = z.shape[0]

        # t 샘플링
        t = torch.randint(0, model.num_timesteps, (b,), device=device).long()
        noise = torch.randn_like(z)

        if model.use_scale:
            # scale이 활성화되었다면 q_sample 수식 구현
            sqrt_alphas_cumprod_t = torch.gather(model.sqrt_alphas_cumprod, 0, t)
            scale_arr_t = torch.gather(model.scale_arr, 0, t)
            sqrt_one_minus_alphas_cumprod_t = torch.gather(model.sqrt_one_minus_alphas_cumprod, 0, t)

            # shape 맞추기
            while sqrt_alphas_cumprod_t.dim() < z.dim():
                sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                scale_arr_t = scale_arr_t.unsqueeze(-1)
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

            x_noisy = sqrt_alphas_cumprod_t * z * scale_arr_t + sqrt_one_minus_alphas_cumprod_t * noise
        else:
            # 기본 q_sample
            x_noisy = model.q_sample(z, t, noise=noise)

        # eps 예측
        eps_pred = model.apply_model(x_noisy, t, cond)

        # loss 계산 (predicted eps vs actual noise)
        loss = loss_fn(eps_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        # 로깅
        if accelerator.is_main_process:
            if global_step % log_interval == 0:
                print(f"Step {global_step}: loss = {loss.item()}")

            # 체크포인트 저장
            if global_step % save_interval == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(save_dir, f"model_{global_step}.pt")
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

    # 마지막 체크포인트 저장
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(save_dir, "model_final.pt")
        torch.save(unwrapped_model.state_dict(), save_path)
        print(f"Training finished. Model saved to {save_path}")


if __name__ == "__main__":
    # 사용자 환경에 맞게 수정
    config_path = "path/to/config.yaml"
    ckpt_path = "path/to/pretrained.ckpt"
    data_root = "latents_data_32s_40frames_vggsound_sparse_new_normalization"
    prompt_file = "path/to/prompts.txt"
    save_dir = "./checkpoints"
    max_steps = 10000
    batch_size = 2
    lr = 1e-4

    train(
        config_path=config_path,
        ckpt_path=ckpt_path,
        data_root=data_root,
        prompt_file=prompt_file,
        save_dir=save_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        lr=lr,
        mixed_precision='fp16' # 필요 시 'no', 'bf16'로 변경
    )
