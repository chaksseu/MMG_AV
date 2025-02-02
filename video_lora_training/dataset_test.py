import os
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import io 
from functools import wraps
from tqdm import tqdm

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
        # self.df = self.df.iloc[valid_indices].reset_index(drop=True)



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        try:
            row = self.df.iloc[idx]
            video_id = row["id"]
            caption_text = row["caption"]

            video_path = os.path.join(self.video_dir, f"{video_id}")
            video, _, info = io.read_video(video_path, pts_unit="sec")

            current_video_frames = video.shape[0]
            if current_video_frames < self.target_frames:
                pad_frames = self.target_frames - current_video_frames
                pad_shape = (pad_frames, video.shape[1], video.shape[2], video.shape[3])
                pad_tensor = torch.zeros(pad_shape, dtype=video.dtype)
                sliced_video = torch.cat([video, pad_tensor], dim=0)
            else:
                max_start = current_video_frames - self.target_frames
                start = random.randint(0, max_start)
                sliced_video = video[start: start + self.target_frames]

            sliced_video = sliced_video.float()
            sliced_video = sliced_video / 127.5 - 1.0
            
            if sliced_video.shape[0] != 40:
                print("video_id:", video_id)
                print("caption:", caption_text)

            return {
                "video_tensor": sliced_video,  # (target_frames, H, W, C)
                "caption": caption_text,
                "video_id": video_id  # 디버깅을 위한 추가 정보
            }
        except Exception as e:
            print(f"[ERROR] 문제 발생 - idx: {idx}, video_id: {row.get('id', 'unknown')}, caption: {row.get('caption', 'unknown')}")
            raise e




def main():
    """
    간단히 Dataset 동작을 확인하고, 에러가 발생한 비디오 파일의 이름을 로깅하기 위한 main 함수 예시
    """

    # 예시용 CSV 경로와 비디오 폴더 경로 (사용 환경에 맞춰 수정)
    csv_path = "/home/jupyter/preprocessed_WebVid_10M_videos_0130.csv"               # 실제 CSV 파일 경로
    video_dir = '/home/jupyter/preprocessed_WebVid_10M_train_videos_0130'  
    split = "train"

    # Dataset 생성
    dataset = VideoTextDataset(
        csv_path=csv_path,
        video_dir=video_dir,
        split=split,
        target_frames=40
    )

    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=32,           # 적절한 배치 사이즈로 설정
        shuffle=False,
        num_workers=32,   
        drop_last=False,
    )


    # tqdm 추가하여 진행률 표시
    for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
        print("video tensor shape", batch["video_tensor"].shape)

if __name__ == "__main__":
    main()