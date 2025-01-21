import os
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL, VaeImageProcessor
from transformers import AutoTokenizer

from preprocess.converter import (
    get_mel_spectrogram_from_audio,
    normalize_spectrogram,
)
from preprocess.utils import pad_spec

from preprocess.auffusion_pipe_functions import (
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
        pretrained_model_name_or_path: str,
        sample_rate: int = 16000,
        slice_duration: float = 3.2,
        hop_size: int = 160,
        n_mels: int = 256,
        pad_to_spec_len: bool = True,
        seed: int = 42,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        # ==============================
        # 1) CSV 로드 및 split 필터링
        # ==============================
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.slice_duration = slice_duration  # 예: 3.2초
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.pad_to_spec_len = pad_to_spec_len
        self.device = device
        self.dtype = dtype

        # (slice_duration 초 * sr / hop_size)로 대략 예상되는 스펙트럼 time축 길이
        self.expected_time_len = int(self.slice_duration * (self.sample_rate / self.hop_size))

        # 랜덤 시드 설정
        self.generator = torch.Generator(device=device).manual_seed(seed)
        random.seed(seed)

        # ================================================
        # 2) 모델/토크나이저/어댑터 등 로딩 (auffusion 예시)
        # ================================================

        # 2-1) pretrained_model_name_or_path가 로컬 폴더가 아니면 snapshot_download
        if not os.path.isdir(pretrained_model_name_or_path):
            self.pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)
        else:
            self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # 2-2) VAE 로드
        with torch.no_grad():
            self.vae = AutoencoderKL.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="vae"
            ).to(self.device, self.dtype)
        self.vae.requires_grad_(False)

        # 2-3) VAE scale factor 기반 ImageProcessor
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        # 2-4) condition_config.json 기반으로 text_encoder_list, tokenizer_list, adapter_list 로딩
        condition_json_path = os.path.join(self.pretrained_model_name_or_path, "condition_config.json")
        with open(condition_json_path, "r", encoding="utf-8") as f:
            condition_json_list = json.load(f)

        self.text_encoder_list = []
        self.tokenizer_list = []
        self.adapter_list = []

        with torch.no_grad():
            for cond_item in condition_json_list:
                # text encoder / tokenizer
                text_encoder_path = os.path.join(self.pretrained_model_name_or_path, cond_item["text_encoder_name"])
                tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
                text_encoder_cls = import_model_class_from_model_name_or_path(text_encoder_path)
                text_encoder = text_encoder_cls.from_pretrained(text_encoder_path).to(self.device, self.dtype)

                tokenizer.requires_grad_(False)
                text_encoder.requires_grad_(False)

                self.tokenizer_list.append(tokenizer)
                self.text_encoder_list.append(text_encoder)

                # condition adapter
                adapter_path = os.path.join(self.pretrained_model_name_or_path, cond_item["condition_adapter_name"])
                adapter = ConditionAdapter.from_pretrained(adapter_path).to(self.device, self.dtype)
                adapter.requires_grad_(False)
                self.adapter_list.append(adapter)


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
        _, spec = get_mel_spectrogram_from_audio(audio_np, device=self.device)
        spec = normalize_spectrogram(spec)  # shape: [n_mels, T]

        # 필요시 pad/crop
        if self.pad_to_spec_len:
            spec = pad_spec(
                spec,
                expected_T=self.expected_time_len,
                pad_value=0.0,
                random_crop=False
            )

        # ============================
        # (4) VAE, Adapter 등 사용
        # ============================
        # 예시 코드 상, spec을 [0,1] 범위 이미지처럼 가정하기 위해 (spec+1)/2 변환
        # 다만 실제로는 spec이 음수가 아닐 수도 있으니, 여기선 예시 그대로 두겠습니다.
        # spec: shape [n_mels, T]
        # 아래처럼 하면 (T, n_mels)별로 잘리는 점 주의 (원코드도 조금 애매합니다).
        # 일단 원 코드 흐름에 맞춰 진행:
        spectrograms = [(row_ + 1) / 2 for row_ in spec]  # list of T개, 각각 shape [n_mels]

        # image_processor 사용 예시
        # 실제 VaeImageProcessor가 어떻게 preprocess하는지는 diffusers 버전에 따라 다를 수 있습니다.
        # 보통 PIL Image나 [B,H,W,C] 텐서를 받는 식이 많으므로,
        # 아래 로직은 실제론 맞지 않을 수 있습니다. (개념적인 예시임)
        image = self.image_processor.preprocess(spectrograms)  # 대략 [1, C, H, W] 형태 반환 가정

        # 텍스트/오디오 프롬프트 인코딩
        with torch.no_grad():
            audio_text_embed = encode_audio_prompt(
                text_encoder_list=self.text_encoder_list,
                tokenizer_list=self.tokenizer_list,
                adapter_list=self.adapter_list,
                tokenizer_model_max_length=77,
                dtype=self.dtype,
                prompt=[caption_text],  # 배치 한 개 예시
                device=self.device
            )

        # VAE 인코딩 -> latents
        with torch.no_grad():
            vae_output = self.vae.encode(image)  # 예: VAEEncoderOutput(latent_dist=...)
            audio_latent = retrieve_latents(vae_output, generator=self.generator)
            # scaling factor 적용
            audio_latent = self.vae.config.scaling_factor * audio_latent

        # 배치 차원 제거 (예: [1, ...] -> [...])
        audio_latent = audio_latent.squeeze(0)
        audio_text_embed = audio_text_embed.squeeze(0)

        return {
            "audio_latent": audio_latent,   # [latent_dim...] 등
            "text_emb": audio_text_embed,   # [hidden_dim]
            "caption": caption_text
        }
