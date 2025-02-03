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


class AudioTextDataset(Dataset):
    """
    CSV 예: [id, caption, split]
      - id: 오디오 파일 이름(확장자 없이)
      - caption: 텍스트 캡션
      - split: "train" 또는 "test"

    audio_dir/{id}.flac 형태로 오디오 파일이 존재한다고 가정.
    최대 30초 이하 오디오이며, 3.2초를 무작위로 슬라이스하여 사용.

    (아래 예시에서는):
    - vae, text_encoder_list, tokenizer_list, adapter_list 등을
      내부에서 직접 로드하는 방식으로 구성.
    - 실제 사용 시에는 외부에서 모델을 주입받는 방식도 가능.
    """

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        split: str,  # "train" or "test"
        sample_rate: int = 16000,
        slice_duration: float = 3.2,
        hop_size: int = 160,
        pad_to_spec_len: bool = True,
    ):
        super().__init__()

        # ==============================
        # 1) CSV 로드 및 split 필터링
        # ==============================
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        #self.df = self.df.head(100)



        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.slice_duration = slice_duration  # 예: 3.2초
        self.hop_size = hop_size
        self.pad_to_spec_len = pad_to_spec_len

        # (slice_duration 초 * sr / hop_size)로 대략 예상되는 스펙트럼 time축 길이
        self.expected_time_len = int(self.slice_duration * (self.sample_rate / self.hop_size))



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_id = row["id"]
        caption_text = row["caption"]

        # ============================
        # (1) 오디오 로딩
        # ============================
        audio_path = os.path.join(self.audio_dir, f"{audio_id}.flac")
        waveform, sr = torchaudio.load(audio_path)

        # resample (필요 시)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # channels x n_samples
        n_samples = waveform.shape[1]
        target_samples = int(self.slice_duration * self.sample_rate)

        # ============================
        # (2) 3.2초 구간 슬라이스
        # ============================
        if n_samples < target_samples:
            # 3.2초보다 짧으면 zero-padding
            pad_size = target_samples - n_samples
            waveform = F.pad(waveform, (0, pad_size), "constant", 0.0)
        else:
            # 무작위 슬라이스
            max_start = n_samples - target_samples
            start = random.randint(0, max_start)
            waveform = waveform[:, start:start + target_samples]

        # (스테레오 -> 모노)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # ============================
        # (3) Mel spectrogram 변환
        # ============================
        audio_np = waveform.squeeze(0).cpu().numpy()  # shape: [samples,]
        # get_mel_spectrogram_from_audio()가 (아마) (None, spec) 형태로 반환된다고 가정
        _, spec = get_mel_spectrogram_from_audio(audio_np)


        spec = normalize_spectrogram(spec)  # shape: [n_mels, T]

        # 필요시 pad/crop
        if self.pad_to_spec_len:
            spec = pad_spec(
                spec,
                spec_length=self.expected_time_len,
                pad_value=0.0,
                random_crop=False
            )

        
        return {
            "spec": spec,   # [latent_dim...] 등
            #"text_emb": audio_text_embed,   # [hidden_dim]
            "caption": caption_text
        }
