import os
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import io 
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class VideoTextDataset(Dataset):
    """
    CSV 예시: [id, caption, split]
      - id: 동영상 파일 이름(확장자 없이)
      - caption: 텍스트 캡션
      - split: "train" 또는 "test"

    video_dir/{id}.mp4 형태로 동영상 파일이 존재한다고 가정.
    최대 프레임 수에서 target_frames(기본 40)을 무작위로 슬라이스하여 사용.
    """

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        split: str,  # "train" 또는 "test"
        target_frames: int = 40,
    ):
        super().__init__()

        # (1) CSV 로드 및 split 필터링
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.video_dir = video_dir
        self.target_frames = target_frames


        # # ---- 파일 유효성 검사 ----
        # valid_indices = []
        # for i in range(len(self.df)):
        #     video_id = self.df.iloc[i]["id"]
        #     video_path = os.path.join(self.video_dir, f"{video_id}")
        #     if not os.path.isfile(video_path):
        #         continue
        #     try:
        #         _, _, _ = io.read_video(video_path, pts_unit="sec")   # video shape: (T, H, W, C)
        #         valid_indices.append(i)
        #     except:
        #         continue
        
        # 손상되지 않은 데이터만 남기기
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        video_id = row["id"]
        caption_text = row["caption"]

        # (2) 비디오 로딩
        video_path = os.path.join(self.video_dir, f"{video_id}")

        video, _, info = io.read_video(video_path, pts_unit="sec")   # video shape: (T, H, W, C)
       

        current_video_frames = video.shape[0]

        # (3) target_frames 개수만큼 프레임 추출
        if current_video_frames < self.target_frames:
            # 여기서는 간단히 뒤가 모자랄 경우 zero padding 예시
            # 필요에 따라 더 적절한 패딩 방식을 구현하세요.
            pad_frames = self.target_frames - current_video_frames
            pad_shape = (pad_frames, video.shape[1], video.shape[2], video.shape[3])
            pad_tensor = torch.zeros(pad_shape, dtype=video.dtype)
            sliced_video = torch.cat([video, pad_tensor], dim=0)
        else:
            # 무작위 슬라이스
            max_start = current_video_frames - self.target_frames
            start = random.randint(0, max_start)
            sliced_video = video[start : start + self.target_frames]

        # (4) 픽셀값 정규화
        # 일반적으로 0~255 범위 -> 0~1 범위: video / 255.0
        # 여기서는 -1~1 범위로 맞춘 예시 (video / 127.5 - 1)
        sliced_video = sliced_video.float()
        before_max = sliced_video.max().item()
        before_min = sliced_video.min().item()

        sliced_video = sliced_video / 127.5 - 1.0

        after_max = sliced_video.max().item()
        after_min = sliced_video.min().item()

        # (5) 디버깅용 출력 (불필요하다면 주석 처리)
        print(f"[DEBUG] Loaded video shape: {video.shape}")
        print(f"[DEBUG] Sliced video shape: {sliced_video.shape}")
        print("[DEBUG] before max:", before_max, "before min:", before_min)
        print("[DEBUG] after max:", after_max, "after min:", after_min)

        return {
            "video_tensor": sliced_video,  # (target_frames, H, W, C)
            "caption": caption_text
        }


def main():
    """
    간단히 Dataset 동작을 확인하기 위한 main 함수 예시
    """

    # 예시용 CSV 경로와 비디오 폴더 경로 (사용 환경에 맞춰 수정)
    csv_path = "/workspace/webvid_download/preprocessed_WebVid_10M_0125_2097.csv"  # 예: id,caption,split
    video_dir = "/workspace/webvid_download/preprocessed_WebVid_10M_train_videos_0125"      # "001.mp4" 등이 들어있는 폴더
    split = "train"

    # Dataset 생성
    dataset = VideoTextDataset(
        csv_path=csv_path,
        video_dir=video_dir,
        split=split,
        target_frames=40
    )

    print(f"Dataset length: {len(dataset)}")

    # 0번 샘플 확인
    if len(dataset) > 0:
        sample = dataset[0]
        video_tensor = sample["video_tensor"]
        caption = sample["caption"]

        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Caption: {caption}")

        sample = dataset[1]
        video_tensor = sample["video_tensor"]
        caption = sample["caption"]

        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Caption: {caption}")

        sample = dataset[2]
        video_tensor = sample["video_tensor"]
        caption = sample["caption"]

        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Caption: {caption}")

        sample = dataset[3]
        video_tensor = sample["video_tensor"]
        caption = sample["caption"]

        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
