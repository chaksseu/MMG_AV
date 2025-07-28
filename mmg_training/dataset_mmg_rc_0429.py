import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io
import torch.nn.functional as F
import numpy as np

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

    [For Random Conditioning]
      - A: ta_csv (text-audio) → ta_df
      - B: tv_csv (text-video) → tv_df
      - C: 메인 CSV → df
      - epoch_length를 세 CSV 중 가장 샘플 수가 많은 데이터셋 기준으로 정함.
      - A, B, C 각각은 자신만의 full cycle(한 바퀴)를 반복하며, full cycle마다 랜덤 순서로 인덱스가 재생성됨.
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
        ta_dir: str = None,
        ta_csv_path: str = None,
        tv_dir: str = None,
        tv_csv_path: str = None,
    ):
        super().__init__()
        # TAV CSV 
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        # TA CSV
        self.ta_df = pd.read_csv(ta_csv_path)
        self.ta_df = self.ta_df[self.ta_df["split"] == split].reset_index(drop=True)
        # TV CSV
        self.tv_df = pd.read_csv(tv_csv_path)
        self.tv_df = self.tv_df[self.tv_df["split"] == split].reset_index(drop=True)
              
        self.spectrogram_dir = spectrogram_dir
        self.video_dir = video_dir

        self.ta_spectrogram_dir = ta_dir
        self.tv_video_dir = tv_dir

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

        # for distillation: epoch 길이를 세 CSV 중 가장 샘플 수가 많은 데이터셋 기준으로 결정
        self.epoch_length = max(len(self.ta_df), len(self.tv_df), len(self.df))
        self.on_epoch_end()  # 초기 페어링 설정

    def _generate_permutation(self, data_length):
        """
        주어진 data_length에 대해, epoch_length 만큼
        full cycle 단위로 랜덤 permutation 인덱스를 생성합니다.
        """
        num_full_cycles = self.epoch_length // data_length
        remainder = self.epoch_length % data_length
        perm = []
        for _ in range(num_full_cycles):
            perm.extend(np.random.permutation(data_length).tolist())
        if remainder:
            perm.extend(np.random.permutation(data_length)[:remainder].tolist())
        return np.array(perm)

    def on_epoch_end(self):
        # A: ta_df, B: tv_df, C: 메인 CSV(df)
        self.permutation_A = self._generate_permutation(len(self.ta_df))
        self.permutation_B = self._generate_permutation(len(self.tv_df))
        self.permutation_C = self._generate_permutation(len(self.df))

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx: int):
        # on_epoch_end()에서 생성된 idx 사용
        main_idx = self.permutation_C[idx]
        ta_idx = self.permutation_A[idx]
        tv_idx = self.permutation_B[idx]

        row = self.df.iloc[main_idx]
        file_id = row["id"]
        # caption_text = row["caption"]
        tav_audio_caption = row["caption"] # llm_audio_caption
        tav_video_caption = row["new_caption"] # llm_video_caption

        ta_row = self.ta_df.iloc[ta_idx]
        ta_file_id = ta_row["id"]
        ta_caption_text = ta_row["caption"]

        tv_row = self.tv_df.iloc[tv_idx]
        tv_file_id = tv_row["id"]
        tv_caption_text = tv_row["caption"]

        # ----- Spectrogram 처리 -----
        spec_path = os.path.join(self.spectrogram_dir, f"{file_id}.pt")
        if not os.path.isfile(spec_path):
            raise FileNotFoundError(f"Spectrogram file not found: {spec_path}")
        spec = torch.load(spec_path)  # shape: [n_mels, total_T]
        spec = normalize_spectrogram(spec)  # 정규화 (값을 [0,1] 등 범위로 변환)

        # ta spec 처리
        ta_spec_path = os.path.join(self.ta_spectrogram_dir, f"{ta_file_id}.pt")
        if not os.path.isfile(ta_spec_path):
            raise FileNotFoundError(f"Spectrogram file not found: {ta_spec_path}")
        ta_spec = torch.load(ta_spec_path)  # shape: [n_mels, total_T]
        ta_spec = normalize_spectrogram(ta_spec)  # 정규화 (값을 [0,1] 등 범위로 변환)


        # ----- Video 처리 -----
        # video 파일 경로 (필요시 파일 확장자 추가 가능)
        video_path = os.path.join(self.video_dir, f"{file_id}.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video, _, _ = io.read_video(video_path, pts_unit="sec")  # shape: (T, H, W, C)

        # tv video 처리
        tv_video_path = os.path.join(self.tv_video_dir, f"{tv_file_id}")
        if not os.path.isfile(tv_video_path):
            raise FileNotFoundError(f"Video file not found: {tv_video_path}")
        tv_video, _, _ = io.read_video(tv_video_path, pts_unit="sec")  # shape: (T, H, W, C)




        ### tv_video slicing ###
        tv_current_video_frames = tv_video.shape[0]
        if tv_current_video_frames < self.target_frames:
            tv_pad_frames = self.target_frames - tv_current_video_frames
            tv_pad_shape = (tv_pad_frames, tv_video.shape[1], tv_video.shape[2], tv_video.shape[3])
            tv_pad_tensor = torch.zeros(tv_pad_shape, dtype=tv_video.dtype)
            tv_video_tensor = torch.cat([tv_video, tv_pad_tensor], dim=0)
        else:
            tv_max_start = tv_current_video_frames - self.target_frames
            tv_start = random.randint(0, tv_max_start)
            tv_video_start = tv_start
            tv_video_tensor = tv_video[tv_start: tv_start + self.target_frames]

        tv_video_tensor = tv_video_tensor.float()
        tv_video_tensor = tv_video_tensor / 127.5 - 1.0  # 픽셀값 정규화

        # 영상의 해상도가 target_height, target_width와 다르면 제로 패딩 적용
        if tv_video_tensor.shape[1] < self.target_height or tv_video_tensor.shape[2] < self.target_width:
            pad_h = max(0,  self.target_height - tv_video_tensor.shape[1])
            pad_w = max(0,  self.target_width - tv_video_tensor.shape[2])
            # (T, H, W, C) -> (T, C, H, W)
            tv_video_tensor = tv_video_tensor.permute(0, 3, 1, 2)
            # F.pad의 패딩 순서는 (pad_left, pad_right, pad_top, pad_bottom)
            tv_video_tensor = F.pad(tv_video_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
            # 다시 (T, H, W, C)로 변환
            tv_video_tensor = tv_video_tensor.permute(0, 2, 3, 1)

        start_h, start_w = 0, 0
        if tv_video_tensor.shape[1] >  self.target_height:
            start_h = random.randint(0, tv_video_tensor.shape[1] -  self.target_height)
        if tv_video_tensor.shape[2] >  self.target_width:
            start_w = random.randint(0, tv_video_tensor.shape[2] -  self.target_width)
        tv_video_tensor = tv_video_tensor[:, start_h:start_h +  self.target_height, start_w:start_w +  self.target_width, :]


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

        if video_tensor.shape[1] < self.target_height or video_tensor.shape[2] < self.target_width:
            _, height, width, _ = video_tensor.shape
            pad_h = max(0, self.target_height - height)
            pad_w = max(0, self.target_width - width)
            video_tensor = video_tensor.permute(0, 3, 1, 2)
            video_tensor = F.pad(video_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
            video_tensor = video_tensor.permute(0, 2, 3, 1)

        start_h, start_w = 0, 0
        if video_tensor.shape[1] > self.target_height:
            start_h = random.randint(0, video_tensor.shape[1] - self.target_height)
        if video_tensor.shape[2] > self.target_width:
            start_w = random.randint(0, video_tensor.shape[2] - self.target_width)
        video_tensor = video_tensor[:, start_h:start_h + self.target_height, start_w:start_w + self.target_width, :]



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


        ### ta_audio slicing ###
        ta_total_T = ta_spec.shape[1]
        if ta_total_T < self.expected_time_len:
            if self.pad_to_spec_len:
                # expected_time_len에 도달할 때까지 제로 패딩 (random_crop=False)
                ta_spec = pad_spec(ta_spec, spec_length=self.expected_time_len, pad_value=0.0, random_crop=False)
        else:
            # 총 T 길이에서 expected_time_len 만큼 랜덤 슬라이스
            ta_max_start = ta_total_T - self.expected_time_len
            # start = random.randint(0, max_start)
            ta_start = tv_video_start * 8 # 320 / 40 = 8
            ta_spec = ta_spec[:, ta_start:ta_start + self.expected_time_len]


        return {
            "spec": spec,                   # 전처리된 spectrogram tensor
            "video_tensor": video_tensor,   # 전처리된 video tensor
            "tav_audio_caption": tav_audio_caption,
            "tav_video_caption": tav_video_caption,
            "id": file_id,                   # 디버깅 및 추후 활용을 위한 id 정보
            "tv_video_tensor": tv_video_tensor,
            "tv_caption": tv_caption_text,
            "ta_spec": ta_spec,
            "ta_caption": ta_caption_text,
        }


# 간단히 Dataset의 동작을 확인하는 main 함수 예시
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # 예시 경로 (사용 환경에 맞게 수정)
    csv_path = "/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs/llm_combined_vgg_csv_0404.csv"
    spectrogram_dir = "/home/work/kby_hgh/workspace/data/preprocessed_VGGSound_train_dataset_0318/preprocessed_VGGSound_train_spec_0310"
    video_dir = "/home/work/kby_hgh/workspace/data/preprocessed_VGGSound_train_no_crop_videos_0329"
    # ta_dir = "/home/work/kby_hgh/workspace/data/MMG_TA_dataset_audiocaps_wavcaps_spec_0320"
    # ta_csv_path = "/home/work/kby_hgh/again_mmg_TA_dataset_zip_0326/MMG_TA_dataset_filtered_0321.csv" 
    # tv_dir = "/home/work/kby_hgh/processed_OpenVid_1M_videos"
    # tv_csv_path = "/home/work/kby_hgh/0411_processed_Openvid_train.csv"
    ta_dir = "/home/work/kby_hgh/workspace/data/preprocessed_VGGSound_train_dataset_0318/preprocessed_VGGSound_train_spec_0310"
    ta_csv_path = "/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs/llm_combined_vgg_csv_0404.csv"
    tv_dir = "/home/work/kby_hgh/workspace/data/preprocessed_VGGSound_train_no_crop_videos_0329"
    tv_csv_path = "/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs/llm_combined_vgg_csv_0404.csv"

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
        ta_dir = ta_dir,
        ta_csv_path = ta_csv_path,
        tv_dir = tv_dir,
        tv_csv_path = tv_csv_path,
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8)

    for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
        print("TAV_Spectrogram shape:", batch["spec"].shape)
        print("TAV_Video tensor shape:", batch["video_tensor"].shape)
        print("TA_Spectrogram shape:", batch["ta_spec"].shape)
        print("TV_Video tensor shape:", batch["tv_video_tensor"].shape)

        print("TAV_Audio_Captions:", batch["tav_audio_caption"])
        print("TAV_Video_Captions:", batch["tav_video_caption"])
        print("TA_Captions:", batch["ta_caption"])
        print("TV_Captions:", batch["tv_caption"])

        # print("IDs:", batch["id"])
