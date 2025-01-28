import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# pad_spec 등 유틸
from preprocess.utils import pad_spec



from preprocess.converter_copy_0123 import (
    normalize_spectrogram,
)

class AudioTextDataset(Dataset):
    """
    - CSV 예시: [id, caption, split]
    - spectrogram_dir/{id}.pt 로 저장된 mel 스펙트럼을 로드.
    - slice_duration 만큼 T축에서 랜덤 슬라이싱 및 패딩.
    """

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        split: str,        # "train" or "test"
        slice_duration: float = 3.2,
        sample_rate: int = 16000,
        hop_size: int = 160,
        pad_to_spec_len: bool = True,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.spectrogram_dir = audio_dir

        self.slice_duration = slice_duration
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.pad_to_spec_len = pad_to_spec_len

        # 3.2초라면, time축 길이는 대략 slice_duration * (sample_rate / hop_size)
        self.expected_time_len = int(self.slice_duration * (self.sample_rate / self.hop_size))

        # # ---- 파일 유효성 검사 ----
        # valid_indices = []
        # for i in range(len(self.df)):
        #     audio_id = self.df.iloc[i]["id"]
        #     spec_path = os.path.join(self.spectrogram_dir, f"{audio_id}.pt")
        #     if not os.path.isfile(spec_path):
        #         continue
        #     try:
        #         _ = torch.load(spec_path)
        #         valid_indices.append(i)
        #     except:
        #         continue
        
        # # 손상되지 않은 데이터만 남기기
        # self.df = self.df.iloc[valid_indices].reset_index(drop=True)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_id = row["id"]
        caption_text = row["caption"]

        # 저장된 Mel 스펙트럼 로드 (shape: [n_mels, total_T])
        spec_path = os.path.join(self.spectrogram_dir, f"{audio_id}.pt")
        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"Spectrogram file not found: {spec_path}")

        spec = torch.load(spec_path)  # e.g. shape [n_mels, total_T]

        spec = normalize_spectrogram(spec)  # shape: [n_mels, T]

        # total_T가 expected_time_len보다 작으면 패딩, 더 크면 랜덤 슬라이싱
        total_T = spec.shape[1]

        if total_T < self.expected_time_len:
            # Zero-padding
            if self.pad_to_spec_len:
                spec = pad_spec(spec, spec_length=self.expected_time_len, pad_value=0.0, random_crop=False)
        else:
            # 3.2초 해당하는 길이만큼 T축 랜덤 슬라이싱
            max_start = total_T - self.expected_time_len
            start = random.randint(0, max_start)
            spec = spec[:, start:start + self.expected_time_len]

        return {
            "audio_latent": spec,     # 실제론 latent가 아니라 mel spec
            "caption": caption_text
        }
