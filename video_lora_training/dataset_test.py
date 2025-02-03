import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from tqdm import tqdm

class VideoTextDataset(Dataset):
    """
    CSV 예시: [id, caption, split]
      - id: 동영상 파일 이름 (확장자 없이)
      - caption: 텍스트 캡션
      - split: "train" 또는 "test"

    동영상 파일은 video_dir/{id}.mp4 로 존재한다고 가정합니다.
    최종적으로 동영상 텐서는 (40, 320, 512, 3) 크기를 갖도록 처리되며,
    만약 공간 해상도가 (320, 512)가 아니라면, 해당 video id와 caption을 로깅합니다.
    """
    def __init__(self, csv_path: str, video_dir: str, split: str, target_frames: int = 40):
        super().__init__()
        # CSV 로드 및 split 필터링
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.video_dir = video_dir
        self.target_frames = target_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        try:
            row = self.df.iloc[idx]
            video_id = row["id"]
            caption_text = row["caption"]

            # 동영상 파일 경로 (CSV에 확장자 없이 id가 기록되어 있다고 가정하여 .mp4 추가)
            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
            video, _, _ = io.read_video(video_path, pts_unit="sec")  # shape: (T, H, W, C)
            print("video.shape", video.shape)

            # -----------------------------------------------------------------
            # (1) Temporal processing: 프레임 수를 target_frames(40)로 고정
            # -----------------------------------------------------------------
            current_frames = video.shape[0]
            if current_frames < self.target_frames:
                # target_frames보다 짧으면 부족한 프레임 수만큼 zero-padding
                pad_frames = self.target_frames - current_frames
                pad_tensor = torch.zeros(
                    (pad_frames, video.shape[1], video.shape[2], video.shape[3]),
                    dtype=video.dtype
                )
                video = torch.cat([video, pad_tensor], dim=0)
            else:
                # target_frames 길이만큼 랜덤 슬라이스
                max_start = current_frames - self.target_frames
                start = random.randint(0, max_start)
                video = video[start: start + self.target_frames]

            # -----------------------------------------------------------------
            # (2) Spatial processing: (H, W)를 (320, 512)로 맞추기
            # -----------------------------------------------------------------
            # 먼저 float 변환 (정규화를 위해)
            video = video.float()

            H, W = video.shape[1], video.shape[2]
            # if H != 320 or W != 512:
            #     print(f"[WARNING] video_id: {video_id}, caption: {caption_text}")
            #     print(f"         원본 해상도: ({H}, {W}) -> (320, 512)로 리사이즈합니다.")
            #     # 영상 텐서의 shape: (T, H, W, C) 이므로, interpolate를 위해 (T, C, H, W)로 변환
            #     video = video.permute(0, 3, 1, 2)  # shape: (T, C, H, W)
            #     video = F.interpolate(video, size=(320, 512), mode='bilinear', align_corners=False)
            #     video = video.permute(0, 2, 3, 1)  # shape: (T, 320, 512, C)

            # -----------------------------------------------------------------
            # (3) 정규화: 픽셀 값을 [0,255] 범위에서 [-1,1] 범위로 변환
            # -----------------------------------------------------------------
            video = video / 127.5 - 1.0

            # -----------------------------------------------------------------
            # (4) 최종 shape 검증: (40, 320, 512, 3)
            # -----------------------------------------------------------------
            if video.shape != (self.target_frames, 320, 512, 3):
                print(f"[ERROR] video_id: {video_id}, caption: {caption_text}")
                print(f"        최종 텐서 shape: {video.shape} (예상: ({self.target_frames}, 320, 512, 3))")

            return {
                "video_tensor": video,  # shape: (40, 320, 512, 3)
                "caption": caption_text,
                "video_id": video_id
            }
        except Exception as e:
            # 예외 발생 시 video id와 caption을 로깅한 후 에러 재발생
            print(f"[ERROR] 문제 발생 - idx: {idx}, video_id: {row.get('id', 'unknown')}, caption: {row.get('caption', 'unknown')}")
            raise e

# -------------------------------------------------------------------
# (1) 사용자 정의 collate_fn 함수
# -------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    batch: list of dict
      - 각 dict에는 "video_tensor", "caption", "video_id"가 포함되어 있음.
      - video_tensor의 shape은 모두 (40, 320, 512, 3)로 동일하다고 가정.
    단순히 각 동영상 텐서를 stack합니다.
    """
    video_tensors = [sample["video_tensor"] for sample in batch]
    captions = [sample["caption"] for sample in batch]
    video_ids = [sample["video_id"] for sample in batch]
    video_tensor_batch = torch.stack(video_tensors, dim=0)  # shape: (B, 40, 320, 512, 3)
    return {
        "video_tensor": video_tensor_batch,
        "caption": captions,
        "video_id": video_ids
    }

def main():
    # 예시용 CSV 경로와 비디오 폴더 경로 (사용 환경에 맞게 수정)
    csv_path = "/home/jupyter/preprocessed_WebVid_10M_videos_0130.csv"
    video_dir = "/home/jupyter/preprocessed_WebVid_10M_train_videos_0130"
    split = "train"

    # Dataset 생성
    dataset = VideoTextDataset(
        csv_path=csv_path,
        video_dir=video_dir,
        split=split,
        target_frames=40
    )

    # DataLoader 생성 (custom_collate_fn 적용)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=32,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    # tqdm을 이용한 진행률 표시
    for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
        # batch["video_tensor"]: shape (B, 40, 320, 512, 3)
        print("video tensor shape:", batch["video_tensor"].shape)
        # 추가 처리 (모델 입력, 디버깅 등)을 여기에 작성

if __name__ == "__main__":
    main()
