import os, re
import glob
import random
import pickle
from typing import List, Tuple, Optional, Iterator, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from moviepy.editor import AudioFileClip
from mpi4py import MPI

import math
import torch.nn.functional as F
from PIL import Image

# converter.py에서 필요한 함수 및 클래스 임포트
from converter import (
    get_mel_spectrogram_from_audio,
    normalize_spectrogram,
    MAX_WAV_VALUE
)
from utils import pad_spec, normalize, denormalize

# 유틸리티 함수 정의
import os
import re
from typing import Optional



import csv

def _extract_category_from_filename(filename: str, youtube_id_to_category: Dict[str, str]) -> Optional[str]:
    """
    파일 이름에서 youtube_id를 추출하고, 매핑 딕셔너리에서 category를 검색합니다.
    
    :param filename: 비디오 파일 이름.
    :param youtube_id_to_category: youtube_id와 category의 매핑 딕셔너리.
    :return: 추출된 category 또는 None.
    """
    base = os.path.basename(filename)
    # 파일명 패턴: vggsoundsync_<youtube_id>_<start_seconds>.mp4
    pattern = r'^vggsoundsync_([A-Za-z0-9_-]+)_(\d+)_fixed\.mp4'

    match = re.match(pattern, base)

    if match:
        youtube_id = match.group(1)
    else:
        print(f"파일 이름 형식이 예상과 다릅니다: {base}")
        return None

    # 매핑 딕셔너리에서 category 검색
    category = youtube_id_to_category.get(youtube_id, None)
    if category:
        return category
    else:
        print(f"youtube_id에 해당하는 category를 찾을 수 없습니다: {youtube_id}")
        return None

def create_youtube_id_to_category_map(csv_path: str) -> Dict[str, str]:
    """
    CSV 파일을 읽어 youtube_id와 category의 매핑 딕셔너리를 생성합니다.
    
    :param csv_path: CSV 파일 경로.
    :return: youtube_id를 키로, category를 값으로 하는 딕셔너리.
    """
    mapping = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                mapping[row['youtube_id']] = row['category']
    except FileNotFoundError:
        print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    except KeyError as e:
        print(f"CSV 파일에서 예상 키를 찾을 수 없습니다: {e}")
    return mapping



# MultimodalDataset 클래스 정의

class MultimodalDataset(Dataset):
    """
    비디오, 오디오(스펙트로그램) 및 텍스트 데이터를 함께 처리하는 데이터셋 클래스.

    :param video_size: 비디오 프레임의 크기 [F, C, H, W].
    :param audio_size: 오디오 스펙트로그램의 크기 [C, N_MELS, T].
    :param video_clips: 비디오 클립의 메타데이터.
    :param video_to_text: 비디오 파일 경로를 텍스트(캡션)로 매핑한 딕셔너리.
    :param shard: 현재 프로세스의 샤드 인덱스.
    :param num_shards: 전체 샤드의 수.
    :param random_flip: True이면 비디오 프레임을 랜덤 수평 뒤집기 적용.
    :param audio_fps: 오디오의 샘플링 레이트.
    :param device: 오디오 처리 시 사용할 디바이스 ('cpu' 또는 'cuda').
    """

    def __init__(
        self,
        video_size: List[int],
        audio_size: List[int],
        video_clips: VideoClips,
        video_to_text: Dict[str, str],
        shard: int = 0,
        num_shards: int = 1,
        random_flip: bool = True,
        audio_fps: Optional[int] = 16000,
        device: str = 'cpu'
    ):
        super().__init__()
        self.video_size = video_size  # [F, C, H, W]
        self.audio_size = audio_size  # [C, N_MELS, T]
        self.random_flip = random_flip
        self.video_clips = video_clips
        self.video_to_text = video_to_text
        self.audio_fps = audio_fps
        self.device = device
        self.size = self.video_clips.num_clips()
        self.shard = shard
        self.num_shards = num_shards

        # 샤딩된 인덱스 생성
        self.indices = list(range(self.size))[shard::num_shards]
        random.shuffle(self.indices)

        # 데이터 증강 변환 정의
        self.transform = self._get_transforms()

    def _get_transforms(self) -> T.Compose:
        p = 0.5 if self.random_flip else 0.0
        return T.Compose([
            T.RandomHorizontalFlip(p),
            T.Resize((self.video_size[2], self.video_size[3]), interpolation=InterpolationMode.BICUBIC),
            #T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = self.indices[idx]
        video, audio_spec, text = self.get_item(actual_idx)
        text = text.replace('_', ' ')
        return {
            'video': video,
            'audio': audio_spec,
            'text': text
        }

    def get_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        while True:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)
                break
            except Exception as e:
                print(f"클립 {idx} 로딩 중 오류 발생: {e}. 다음 클립으로 이동합니다.")
                idx = (idx + 1) % self.size

        # 비디오 처리
        video = self._adjust_video_length(video)
        video_after_process = self.process_video(video)
        video_after_process = video_after_process.float() / 127.5 - 1  # -1 - 1
        
        #video = video.permute([0,3,1,2])
        #video_after_process = self.transform(video)
        


        # 오디오 처리
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        duration_per_frame = self.video_clips.video_pts[video_idx][1] - self.video_clips.video_pts[video_idx][0]
        video_fps = self.video_clips.video_fps[video_idx]
        audio_fps = self.audio_fps if self.audio_fps  else info['audio_fps']

        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        clip_pid = clip_pts // duration_per_frame
    
        start_t = (clip_pid[0] / video_fps * 1. ).item()
        end_t = ((clip_pid[-1] + 1) / video_fps * 1. ).item()
       
        video_path = self.video_clips.video_paths[video_idx]
        raw_audio =  AudioFileClip(video_path, fps=audio_fps).subclip(start_t, end_t) 
        raw_audio= raw_audio.to_soundarray()

        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)  # 모노 채널로 변환

        audio_spec = self._process_audio(audio_clip = raw_audio)
        
        # 텍스트 처리
        video_path = self.video_clips.video_paths[video_idx]
        text = self.video_to_text.get(os.path.basename(video_path), "")

        return video_after_process, audio_spec, text

    def _adjust_video_length(self, video: torch.Tensor) -> torch.Tensor:
        """
        비디오 프레임 수를 조정하여 고정된 길이를 맞춥니다.

        :param video: 원본 비디오 텐서 [F, C, H, W].
        :return: 조정된 비디오 텐서 [F, C, H, W].
        """
        desired_length = self.video_size[0]
        current_length = video.shape[0]

        if current_length < desired_length:
            pad_length = desired_length - current_length
            pad_frames = video[-1:].repeat((pad_length, 1, 1, 1))
            video = torch.cat([video, pad_frames], dim=0)
        else:
            video = video[:desired_length]

        return video

    def _process_audio(self, audio_clip=None) -> torch.Tensor:
        audio_data = audio_clip
        expected_T = self.audio_size[-1]

        _, spec = get_mel_spectrogram_from_audio(audio_data, device=self.device)
        spec = normalize_spectrogram(spec)
        spec = pad_spec(spec, expected_T, pad_value=0.0, random_crop=True)
        spec = normalize(spec)
        return spec
    
    def process_video(self, video):
        '''
        resize img to target_size with padding, 
        augment with RandomHorizontalFlip if self.random_flip is True.

        :param video: ten[f, c, h, w]
        ''' 
        video = video.permute([0,3,1,2])
        '''
        old_size = video.shape[2:4]
        ratio = min(float(self.video_size[2])/(old_size[0]), float(self.video_size[3])/(old_size[1]) )
        new_size = tuple([int(i*ratio) for i in old_size])
        pad_w = self.video_size[3] - new_size[1]
        pad_h = self.video_size[2]- new_size[0]
        top,bottom = pad_h//2, pad_h-(pad_h//2)
        left,right = pad_w//2, pad_w -(pad_w//2)
        
        transform = T.Compose([T.RandomHorizontalFlip(self.random_flip), 
                               T.Resize(new_size, interpolation=InterpolationMode.BICUBIC), 
                               T.Pad((left, top, right, bottom))])
        '''
        p = 0.5 if self.random_flip else 0.0

        transform = T.Compose([
            T.RandomHorizontalFlip(p),
            T.Resize((self.video_size[2], self.video_size[3]), interpolation=InterpolationMode.BICUBIC),
            #T.ToTensor(),
            #T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        video_new = transform(video)
        return video_new

    def __del__(self):
        # 비디오 리더 닫기
        if hasattr(self, 'video_clips') and self.video_clips:
            try:
                self.video_clips._video_reader.close()  # 내부 비디오 리더 닫기
            except AttributeError:
                pass  # 비디오 리더가 없다면 무시

# 데이터 로드 함수 정의

def load_data(
    data_dir: str,
    batch_size: int,
    video_size: List[int],
    audio_size: List[int],
    random_flip: bool = True,
    num_workers: int = 4,
    video_fps: int = 64,
    audio_fps: Optional[int] = 16000,
    frame_gap: int = 1,
    drop_last: bool = True,
    device: str = 'cpu'
) -> Iterator[Dict[str, torch.Tensor]]:
    """
    데이터셋 디렉토리에서 비디오-오디오-텍스트 페어를 생성하는 제너레이터를 반환합니다.

    :return: 비디오-오디오-텍스트 페어를 생성하는 제너레이터.
    """
    if not data_dir:
        raise ValueError("데이터 디렉토리가 지정되지 않았습니다.")
    


    csv_path = 'Dataset/vggsoundsync/vggsoundsync.csv'
    youtube_id_to_category = create_youtube_id_to_category_map(csv_path)



    # 비디오 파일 목록 수집
    all_files = _list_video_files_recursively(data_dir)
    
    # 'train' split만 필터링
    all_files = [f for f in all_files if '_fixed.mp4' in f]

    # 파일에서 category 추출하여 매핑 생성
    video_to_text = {}
    for f in all_files:
        category = _extract_category_from_filename(f, youtube_id_to_category)
        if category:
            video_to_text[os.path.basename(f)] = category
        else:
            print(f"파일에서 category를 추출할 수 없습니다: {f}")

    print(f"비디오 파일 개수 (train): {len(all_files)}")

    # MPI 초기화
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    clip_length_in_frames = video_size[0]
    frames_between_clips = frame_gap
    meta_fname = os.path.join(
        data_dir,
        f"video_clip_f{clip_length_in_frames}_g{frames_between_clips}_r{video_fps}.pkl"
    )

    if not os.path.exists(meta_fname):
        if rank == 0:
            print(f"{meta_fname} 파일을 준비 중...")

        video_clips = VideoClips(
            video_paths=all_files,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            num_workers=num_workers,
            frame_rate=video_fps
        )

        if rank == 0:
            with open(meta_fname, 'wb') as f:
                pickle.dump(video_clips.metadata, f)
    else:
        if rank == 0:
            print(f"{meta_fname} 파일을 로드 중...")
        with open(meta_fname, 'rb') as f:
            metadata = pickle.load(f)

        video_clips = VideoClips(
            video_paths=all_files,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            frame_rate=video_fps,
            _precomputed_metadata=metadata
        )

    if rank == 0:
        print(f"{meta_fname}에서 {video_clips.num_clips()}개의 비디오 클립 로드 완료.")

    dataset = MultimodalDataset(
        video_size=video_size,
        audio_size=audio_size,
        video_clips=video_clips,
        video_to_text=video_to_text,
        shard=rank,
        num_shards=size,
        random_flip=random_flip,
        audio_fps=audio_fps,
        device=device
    )

    print("dataset lenght", dataset.__len__())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    #while True:
    #    for batch in loader:
    #        yield batch
    return loader


def _list_video_files_recursively(data_dir: str) -> List[str]:
    """
    주어진 디렉토리에서 비디오 파일을 재귀적으로 탐색하여 리스트로 반환합니다.

    :param data_dir: 탐색할 디렉토리 경로.
    :return: 비디오 파일의 전체 경로 리스트.
    """
    video_extensions = ("*.avi", "*.gif", "*.mp4", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return sorted(video_files)

# 메인 함수 정의

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description="멀티모달 데이터 로더 예제")
        parser.add_argument('--data_dir', type=str, default='Dataset/vggsoundsync/vggsoundsync', help='비디오 파일들이 저장된 데이터셋 디렉토리 경로')
        parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
        parser.add_argument('--seconds', type=float, default=3.2, help='클립 길이(초)')
        parser.add_argument('--video_fps', type=int, default=12.5, help='비디오 프레임 속도')
        parser.add_argument('--audio_fps', type=int, default=16000, help='오디오 샘플링 속도')
        parser.add_argument('--image_resolution', type=int, default=256, help='이미지 해상도')
        parser.add_argument('--frame_gap', type=int, default=40, help='프레임 간격')
        parser.add_argument('--random_flip', action='store_true', help='랜덤 수평 뒤집기 사용')
        parser.add_argument('--device', type=str, default='cpu', help='사용할 디바이스 (cpu 또는 cuda)')
        return parser.parse_args()

    args = parse_args()



    # 비디오 및 오디오 크기 설정
    clip_length_in_seconds = args.seconds
    video_fps = args.video_fps
    F = int(clip_length_in_seconds * video_fps)  # 프레임 수

    video_size = [
        F,                                # F
        3,                                # C
        args.image_resolution,            # H
        args.image_resolution             # W
    ]

    # 오디오 스펙트로그램 크기 설정
    hop_size = 160  # hop_size는 스펙트로그램 변환 시 사용된 파라미터와 일치해야 합니다.
    T_spec = int(clip_length_in_seconds * (args.audio_fps / hop_size))  # 스펙트로그램의 시간 축 크기
    num_mels = 256

    audio_size = [
        3,          # C (normalize_spectrogram 함수에서 3채널로 반환)
        num_mels,   # N_MELS
        T_spec      # T
    ]

    data_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        video_size=video_size,
        audio_size=audio_size,
        frame_gap=args.frame_gap,
        random_flip=args.random_flip,
        num_workers=64,
        video_fps=video_fps,
        audio_fps=args.audio_fps,
        device=args.device
    )

    '''
    from tqdm import tqdm

    def calculate_dataset_statistics(data_loader):
        """
        DataLoader 전체 데이터를 순회하며 max, min, mean, std를 계산합니다.
        """
        video_max, video_min = float('-inf'), float('inf')
        video_sum, video_sum_sq = 0.0, 0.0
        video_count = 0

        audio_max, audio_min = float('-inf'), float('inf')
        audio_sum, audio_sum_sq = 0.0, 0.0
        audio_count = 0

        for batch in tqdm(data_loader, desc="Processing batches"):
            # 비디오와 오디오 데이터 가져오기
            batch_video = batch['video'].detach().cpu()  # [N, F, C, H, W]
            batch_audio = batch['audio'].detach().cpu()  # [N, C, H, W]

            # 배치 크기
            num_video_elements = batch_video.numel()
            num_audio_elements = batch_audio.numel()

            # 비디오 max/min 업데이트
            video_max = max(video_max, batch_video.max().item())
            video_min = min(video_min, batch_video.min().item())

            # 비디오 합계 및 제곱합
            video_sum += batch_video.sum().item()
            video_sum_sq += (batch_video**2).sum().item()
            video_count += num_video_elements

            # 오디오 max/min 업데이트
            audio_max = max(audio_max, batch_audio.max().item())
            audio_min = min(audio_min, batch_audio.min().item())

            # 오디오 합계 및 제곱합
            audio_sum += batch_audio.sum().item()
            audio_sum_sq += (batch_audio**2).sum().item()
            audio_count += num_audio_elements

        # 비디오 평균 및 표준편차 계산
        video_mean = video_sum / video_count
        video_std = ((video_sum_sq / video_count) - (video_mean**2))**0.5

        # 오디오 평균 및 표준편차 계산
        audio_mean = audio_sum / audio_count
        audio_std = ((audio_sum_sq / audio_count) - (audio_mean**2))**0.5

        # 결과 출력
        print("=== Dataset Statistics ===")
        print(f"Video - max: {video_max}, min: {video_min}, mean: {video_mean:.4f}, std: {video_std:.4f}")
        print(f"Audio - max: {audio_max}, min: {audio_min}, mean: {audio_mean:.4f}, std: {audio_std:.4f}")


    # 실행
    calculate_dataset_statistics(data_loader)

    
    '''


    try:
        for batch_idx, batch in enumerate(data_loader, 1):
            batch_video = batch['video']  # [N, F, C, H, W]
            batch_audio = batch['audio']  # [N, C, N_MELS, T]
            batch_text = batch['text']    # [N]
            print(f"\n=== 배치 {batch_idx} ===")
            print(f"batch_video shape: {batch_video.shape}")
            print(f"batch_audio shape: {batch_audio.shape}")
            
            # batch_video 통계 출력
            video_max = batch_video.max().item()
            video_min = batch_video.min().item()
            video_mean = batch_video.mean().item()
            video_std = batch_video.std().item()
            print(f"batch_video - max: {video_max}, min: {video_min}, mean: {video_mean}, std: {video_std}")

            # batch_audio 통계 출력
            audio_max = batch_audio.max().item()
            audio_min = batch_audio.min().item()
            audio_mean = batch_audio.mean().item()
            audio_std = batch_audio.std().item()
            print(f"batch_audio - max: {audio_max}, min: {audio_min}, mean: {audio_mean}, std: {audio_std}")

            # 여기서 모델 학습이나 기타 처리를 수행할 수 있습니다.
    except KeyboardInterrupt:
        print("데이터 로딩 중단.")
    