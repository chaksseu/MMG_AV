import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io
import torch.nn.functional as F

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


# spectrogram 전처리 유틸리티
from preprocess.utils import pad_spec
from preprocess.converter_copy_0123 import normalize_spectrogram

class AudioVideoDataset(Dataset):
    """
    CSV 예시: [id, caption, split]
      - id: 파일 이름 (확장자 없이 혹은 포함; spectrogram은 {id}.pt, video는 {id} 또는 {id}.mp4)
      - caption: 텍스트 캡션
      - split: "train" 또는 "test"

    [Spectrogram 처리]
      - spectrogram_dir/{id}.pt 파일 로드
      - normalize_spectrogram() 함수를 통해 정규화
      - slice_duration(초) 길이에 해당하는 time 축 길이만큼 랜덤 슬라이싱 혹은 패딩

    [Video 처리]
      - video_dir/{id} 파일 로드 (필요시 파일 확장자 추가)
      - 전체 프레임 수가 target_frames 보다 작으면 제로 패딩, 크면 랜덤 슬라이싱
      - 영상 픽셀 값을 [−1, 1] 범위로 정규화하며, target_height 및 target_width에 맞게 패딩 처리
    """
    def __init__(
        self,
        csv_path: str,
        spectrogram_dir: str,
        video_dir: str,
        split: str,  # "train" 또는 "test"
        slice_duration: float = 3.2,
        sample_rate: int = 16000,
        hop_size: int = 160,
        pad_to_spec_len: bool = True,
        target_frames: int = 40,
        target_height: int = 320,
        target_width: int = 512,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.spectrogram_dir = spectrogram_dir
        self.video_dir = video_dir

        # Spectrogram 관련 파라미터
        self.slice_duration = slice_duration
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.pad_to_spec_len = pad_to_spec_len
        self.expected_time_len = int(self.slice_duration * (self.sample_rate / self.hop_size)) # 예: 3.2초 -> expected_time_len = 3.2 * (16000/160) = 320

        # Video 관련 파라미터
        self.target_frames = target_frames
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_id = row["id"]
        caption_text = row["caption"]

        # ----- Spectrogram 처리 -----
        spec_path = os.path.join(self.spectrogram_dir, f"{file_id}.pt")
        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"Spectrogram file not found: {spec_path}")
        spec = torch.load(spec_path)  # shape: [n_mels, total_T]
        spec = normalize_spectrogram(spec)  # 정규화 (값을 [0,1] 등 범위로 변환)


        # ----- Video 처리 -----
        # video 파일 경로 (필요시 파일 확장자 추가 가능)
        video_path = os.path.join(self.video_dir, f"{file_id}.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video, _, _ = io.read_video(video_path, pts_unit="sec")  # shape: (T, H, W, C)




        ### video slicing ###
        current_video_frames = video.shape[0]
        if current_video_frames < self.target_frames:
            pad_frames = self.target_frames - current_video_frames
            pad_shape = (pad_frames, video.shape[1], video.shape[2], video.shape[3])
            pad_tensor = torch.zeros(pad_shape, dtype=video.dtype)
            video_tensor = torch.cat([video, pad_tensor], dim=0)
        else:
            max_start = current_video_frames - self.target_frames
            start = random.randint(0, max_start)
            video_start = start
            video_tensor = video[start: start + self.target_frames]

        video_tensor = video_tensor.float()
        video_tensor = video_tensor / 127.5 - 1.0  # 픽셀값 정규화

        # 영상의 해상도가 target_height, target_width와 다르면 제로 패딩 적용
        if video_tensor.shape[1] != self.target_height or video_tensor.shape[2] != self.target_width:
            print("video_id:", file_id)
            print("caption:", caption_text)
            print("sliced_video.shape before padding:", video_tensor.shape)

            _, height, width, _ = video_tensor.shape
            pad_h = max(0, self.target_height - height)
            pad_w = max(0, self.target_width - width)
            # F.pad의 pad 순서는 (left, right, top, bottom, front, back) 순서로 적용됨
            video_tensor = F.pad(video_tensor, (0, 0, 0, pad_w, 0, pad_h), mode="constant", value=0)

            print("sliced_video.shape after padding:", video_tensor.shape)


        ### audio slicing ###
        total_T = spec.shape[1]
        if total_T < self.expected_time_len:
            if self.pad_to_spec_len:
                # expected_time_len에 도달할 때까지 제로 패딩 (random_crop=False)
                spec = pad_spec(spec, spec_length=self.expected_time_len, pad_value=0.0, random_crop=False)
        else:
            # 총 T 길이에서 expected_time_len 만큼 랜덤 슬라이스
            max_start = total_T - self.expected_time_len
            # start = random.randint(0, max_start)
            start = video_start * 8 # 320 / 40 = 8
            spec = spec[:, start:start + self.expected_time_len]




        return {
            "spec": spec,                   # 전처리된 spectrogram tensor
            "video_tensor": video_tensor,   # 전처리된 video tensor
            "caption": caption_text,
            "id": file_id                   # 디버깅 및 추후 활용을 위한 id 정보
        }


# 간단히 Dataset의 동작을 확인하는 main 함수 예시
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # 예시 경로 (사용 환경에 맞게 수정)
    csv_path = "/workspace/vggsound_processing/New_VGGSound_0311.csv"
    spectrogram_dir = "/workspace/data/preprocessed_VGGSound_train_spec_0310"
    video_dir = "/workspace/data/preprocessed_VGGSound_train_videos_0313"
    split = "train"

    dataset = AudioVideoDataset(
        csv_path=csv_path,
        spectrogram_dir=spectrogram_dir,
        video_dir=video_dir,
        split=split,
        slice_duration=3.2,
        sample_rate=16000,
        hop_size=160,
        pad_to_spec_len=True,
        target_frames=40,
        target_height=320,
        target_width=512,
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16)

    for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
        print("Spectrogram shape:", batch["spec"].shape)
        print("Video tensor shape:", batch["video_tensor"].shape)
        print("Captions:", batch["caption"])
        # print("IDs:", batch["id"])
