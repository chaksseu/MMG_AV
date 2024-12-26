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



def _extract_category_from_filename(filename: str) -> Optional[str]:
    """
    파일 이름에서 category를 추출합니다. youtube_id는 숫자로 끝나고, category는 그 뒤에 위치합니다.
    
    :param filename: 비디오 파일 이름.
    :return: 추출된 category 또는 None.
    """
    # 파일 이름에서 기본 이름 추출
    base = os.path.basename(filename)
    
    # 정규식 패턴을 사용하여 youtube_id와 category 추출
    # 파일 이름 형식: vggsound_sparse_{youtube_id}_{category}_{split}_fixed.mp4
    pattern = r'vggsound_sparse_(.+)_(train|test)_fixed\.mp4'
    match = re.match(pattern, base)
    
    if match:
        # youtube_id와 category 부분을 추출 (전체)
        full_id_category = match.group(1)
        split = match.group(2)
        
        # youtube_id는 숫자로 끝난다고 가정하고 마지막 숫자를 기준으로 구분
        # 숫자로 끝나는 youtube_id를 식별
        youtube_id_match = re.search(r'(.+_\d+)', full_id_category)
        
        if not youtube_id_match:
            print(f"youtube_id를 파일 이름에서 찾을 수 없습니다: {filename}")
            return None
        
        # youtube_id가 끝나는 위치까지 잘라서 youtube_id로 추출
        youtube_id = youtube_id_match.group(1)
        category = full_id_category[len(youtube_id)+1:]  # youtube_id 다음의 부분이 category
        
        # 확인용 출력 (옵션)
        print(f"youtube_id: {youtube_id}, category: {category}, split: {split}")
        
        return category
    else:
        print(f"파일 이름 형식이 예상과 다릅니다: {filename}")
        return None


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
        """
        비디오 프레임에 적용할 변환을 정의합니다.

        :return: 변환 파이프라인.
        """
        transforms_list = []
        if self.random_flip:
            transforms_list.append(T.RandomHorizontalFlip())

        transforms_list.extend([
            T.Resize((self.video_size[2], self.video_size[3]), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        return T.Compose(transforms_list)

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
        video_after_process = self.new_process_video(video)
        #video_after_process = self.process_video(video)
        video_after_process = video_after_process.float() / 127.5 - 1  # 0-1

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
        spec = pad_spec(spec, expected_T, pad_value=0.0, random_crop=False) # True
        #spec = normalize(spec)
        return spec
    
    def process_video(self, video):
        '''
        resize img to target_size with padding, 
        augment with RandomHorizontalFlip if self.random_flip is True.

        :param video: ten[f, c, h, w]
        ''' 
        video = video.permute([0,3,1,2])
        old_size = video.shape[2:4]
        ratio = min(float(self.video_size[2])/(old_size[0]), float(self.video_size[3])/(old_size[1]) )
        new_size = tuple([int(i*ratio) for i in old_size])
        pad_w = self.video_size[3] - new_size[1]
        pad_h = self.video_size[2]- new_size[0]
        top,bottom = pad_h//2, pad_h-(pad_h//2)
        left,right = pad_w//2, pad_w -(pad_w//2)
        transform = T.Compose([T.RandomHorizontalFlip(self.random_flip), T.Resize(new_size, interpolation=InterpolationMode.BICUBIC), T.Pad((left, top, right, bottom))])
        video_new = transform(video)
        return video_new
    
    def new_process_video(self, video):
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

    # 비디오 파일 목록 수집
    all_files = _list_video_files_recursively(data_dir)
    
    # 'train' split만 필터링
    ###all_files = [f for f in all_files if '_train_fixed.mp4' in f]
    # split만 필터링
    split = "test"
    all_files = [f for f in all_files if f'_{split}_fixed.mp4' in f]


    # 파일에서 category 추출하여 매핑 생성
    video_to_text = {}
    for f in all_files:
        category = _extract_category_from_filename(f)
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
        f"video_clip_f{clip_length_in_frames}_g{frames_between_clips}_r{video_fps}_{split}.pkl"
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
        parser.add_argument('--data_dir', type=str, default='Dataset/vggsound_sparse/vggsound_sparse', help='비디오 파일들이 저장된 데이터셋 디렉토리 경로')
        parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
        parser.add_argument('--seconds', type=float, default=4, help='클립 길이(초)')
        parser.add_argument('--video_fps', type=int, default=16, help='비디오 프레임 속도')
        parser.add_argument('--audio_fps', type=int, default=16000, help='오디오 샘플링 속도')
        parser.add_argument('--image_resolution', type=int, default=256, help='이미지 해상도')
        parser.add_argument('--frame_gap', type=int, default=64, help='프레임 간격')
        parser.add_argument('--random_flip', action='store_true', help='랜덤 수평 뒤집기 사용')
        parser.add_argument('--device', type=str, default='cuda', help='사용할 디바이스 (cpu 또는 cuda)')
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
        num_workers=4,
        video_fps=video_fps,
        audio_fps=args.audio_fps,
        device=args.device
    )

    try:
        for batch_idx, batch in enumerate(data_loader, 1):
            batch_video = batch['video']  # [N, F, C, H, W]
            batch_audio = batch['audio']  # [N, C, N_MELS, T]
            batch_text = batch['text']    # [N]
            print(f"배치 {batch_idx}: 비디오 텐서 크기 {batch_video.shape}, 오디오 스펙트로그램 크기 {batch_audio.shape}, 텍스트: {batch_text}")
            # 여기서 모델 학습이나 기타 처리를 수행할 수 있습니다.
    except KeyboardInterrupt:
        print("데이터 로딩 중단.")