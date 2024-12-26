import os
import re
import glob
import random
import pickle
import math
from typing import List, Tuple, Optional, Iterator, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from moviepy.editor import AudioFileClip
from mpi4py import MPI
from PIL import Image

# 추가: CSV 파일을 읽기 위해 pandas 사용
import pandas as pd

# Utility functions from external modules
from preprocess.converter import (
    get_mel_spectrogram_from_audio,
    normalize_spectrogram,
    MAX_WAV_VALUE
)
from preprocess.utils import pad_spec, normalize, denormalize


class MultimodalDataset(Dataset):
    """
    A dataset class that handles synchronized video frames, audio (spectrogram), and text data.
    The text is retrieved from a CSV file using the video path as the key.

    Args:
        video_size (List[int]): The desired video frame size [F, C, H, W].
        audio_size (List[int]): The desired audio spectrogram size [C, N_MELS, T].
        video_clips (VideoClips): VideoClips object containing video metadata.
        csv_path (str): Path to a CSV file mapping 'video_path' -> 'text'.
        shard (int, optional): Current process shard index. Defaults to 0.
        num_shards (int, optional): Total number of shards. Defaults to 1.
        random_flip (bool, optional): If True, apply random horizontal flip to video frames. Defaults to True.
        audio_fps (Optional[int], optional): Sampling rate for audio. Defaults to 16000.
        device (str, optional): Device to use for audio processing ('cpu' or 'cuda'). Defaults to 'cpu'.
    """

    def __init__(
        self,
        video_size: List[int],
        audio_size: List[int],
        video_clips: VideoClips,
        csv_path: str,
        shard: int = 0,
        num_shards: int = 1,
        random_flip: bool = True,
        audio_fps: Optional[int] = 16000,
        device: str = 'cpu'
    ):
        super().__init__()
        self.video_size = video_size  # [F, C, H, W]
        self.audio_size = audio_size  # [C, N_MELS, T]
        self.video_clips = video_clips
        self.csv_path = csv_path
        self.shard = shard
        self.num_shards = num_shards
        self.random_flip = random_flip
        self.audio_fps = audio_fps
        self.device = device

        # Total number of clips in VideoClips
        self.size = self.video_clips.num_clips()

        # Create a list of indices for the current shard and shuffle
        self.indices = list(range(self.size))[shard::num_shards]
        random.shuffle(self.indices)

        # Read CSV file and build a mapping from video path to text
        self.video_to_text_map = self._load_text_from_csv(csv_path)

    def _load_text_from_csv(self, csv_path: str) -> Dict[str, str]:
        """
        Load a CSV file that contains columns ['video_path', 'text'].
        Returns a dict mapping video_path -> text.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        video_to_text = {}
        for _, row in df.iterrows():
            # CSV에 'video_path'와 'text'라는 컬럼이 있다고 가정
            vp = row['video_path']
            txt = row['text']
            video_to_text[vp] = txt
        return video_to_text

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single data example consisting of video, audio, and text.

        Args:
            idx (int): Index in the shard-specific list.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'video': Tensor of shape [F, C, H, W]
                - 'audio': Tensor of shape [C, N_MELS, T]
                - 'text': The text string
        """
        actual_idx = self.indices[idx]
        video, audio_spec, text = self.get_item(actual_idx)

        # Text 후처리 (예: "_" -> " " 변경 등)
        text = text.replace('_', ' ')
        
        return {
            'video': video,
            'audio': audio_spec,
            'text': text
        }

    def get_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Internal method to load and process video, audio, and text data.

        Args:
            idx (int): Index in the overall video_clips (not shard-specific).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: Processed video tensor, audio spectrogram, and text.
        """
        # Safely get a video clip (retry if there's an error)
        while True:
            try:
                video, _, info, video_idx = self.video_clips.get_clip(idx)
                break
            except Exception as e:
                print(f"Error loading clip {idx}: {e}. Moving to next clip.")
                idx = (idx + 1) % self.size

        # Adjust the video length if needed
        video = self._adjust_video_length(video)

        # Perform additional transforms
        video_after_process = self._process_video_transform(video)

        # Convert video from [0,255] range to [-1,1] range
        video_after_process = video_after_process.float() / 127.5 - 1.0

        # Get the original video path to load audio
        video_path = self.video_clips.video_paths[video_idx]

        # Retrieve audio
        video_fps = self.video_clips.video_fps[video_idx]
        audio_fps = self.audio_fps if self.audio_fps else info['audio_fps']

        # Clip information
        # 'clip_idx'는 해당 video_path 내에서 몇 번째 clip인지 알려줌
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        duration_per_frame = self.video_clips.video_pts[video_idx][1] - self.video_clips.video_pts[video_idx][0]
        clip_pid = clip_pts // duration_per_frame

        # Calculate subclip start and end times (in seconds)
        start_t = (clip_pid[0] / video_fps * 1.0).item()
        end_t = ((clip_pid[-1] + 1) / video_fps * 1.0).item()

        # Load and process raw audio using moviepy
        raw_audio_clip = AudioFileClip(video_path, fps=audio_fps).subclip(start_t, end_t)
        raw_audio = raw_audio_clip.to_soundarray()

        # Convert to mono if multiple channels
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)

        # Process audio to spectrogram
        audio_spec = self._process_audio(raw_audio)

        # Retrieve corresponding text using video_path
        text = self.video_to_text_map.get(video_path, "No Text Found")

        return video_after_process, audio_spec, text

    def _adjust_video_length(self, video: torch.Tensor) -> torch.Tensor:
        """
        Adjust the number of frames in the video to match the desired length (pad or trim).

        Args:
            video (torch.Tensor): Video tensor of shape [F, H, W, C].

        Returns:
            torch.Tensor: Adjusted video tensor of shape [F, H, W, C].
        """
        desired_length = self.video_size[0]
        current_length = video.shape[0]

        if current_length < desired_length:
            pad_length = desired_length - current_length
            # Pad with the last frame
            pad_frames = video[-1:].repeat((pad_length, 1, 1, 1))
            video = torch.cat([video, pad_frames], dim=0)
        else:
            video = video[:desired_length]

        return video

    def _process_audio(self, audio_clip: np.ndarray) -> torch.Tensor:
        """
        Convert raw audio into a mel-spectrogram, normalize it,
        and pad or trim to the desired length.

        Args:
            audio_clip (np.ndarray): 1D numpy array of audio samples.

        Returns:
            torch.Tensor: Processed mel-spectrogram tensor [C, N_MELS, T].
        """
        expected_T = self.audio_size[-1]
        _, spec = get_mel_spectrogram_from_audio(audio_clip, device=self.device)
        spec = normalize_spectrogram(spec)
        spec = pad_spec(spec, expected_T, pad_value=0.0, random_crop=False)
        return spec

    def _process_video_transform(self, video: torch.Tensor) -> torch.Tensor:
        """
        Custom video transform: random flip + resize to the desired resolution.
        Input shape: [F, H, W, C]
        Output shape: [F, C, H, W]

        Args:
            video (torch.Tensor): Video tensor of shape [F, H, W, C].

        Returns:
            torch.Tensor: Transformed video tensor [F, C, H, W].
        """
        # Permute to [F, C, H, W]
        video = video.permute([0, 3, 1, 2])

        p = 0.5 if self.random_flip else 0.0

        transform = T.Compose([
            T.RandomHorizontalFlip(p),
            T.Resize((self.video_size[2], self.video_size[3]), interpolation=InterpolationMode.BICUBIC),
        ])
        return transform(video)

    def __del__(self):
        """Close the internal video reader when the dataset is destroyed."""
        if hasattr(self, 'video_clips') and self.video_clips:
            try:
                self.video_clips._video_reader.close()
            except AttributeError:
                pass


def _list_video_files_recursively(data_dir: str) -> List[str]:
    """
    Recursively find all video files (avi, gif, mp4, mov, mkv) in a directory.

    Args:
        data_dir (str): Directory path.

    Returns:
        List[str]: Sorted list of full paths to all found video files.
    """
    video_extensions = ("*.avi", "*.gif", "*.mp4", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return sorted(video_files)


def load_data(
    data_dir: str,
    csv_path: str,
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
    Creates a DataLoader for the multimodal dataset (video + audio + text).

    Args:
        data_dir (str): Directory containing video files.
        csv_path (str): Path to CSV file mapping 'video_path' -> 'text'.
        batch_size (int): Batch size.
        video_size (List[int]): [F, C, H, W].
        audio_size (List[int]): [C, N_MELS, T].
        random_flip (bool, optional): Whether to randomly flip video frames. Defaults to True.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        video_fps (int, optional): Video frame rate for reading. Defaults to 64.
        audio_fps (Optional[int], optional): Audio sampling rate. Defaults to 16000.
        frame_gap (int, optional): Gap between frames in the clip. Defaults to 1.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        device (str, optional): Device to use for audio processing. Defaults to 'cpu'.

    Returns:
        Iterator[Dict[str, torch.Tensor]]: A DataLoader producing batches of multimodal data.
    """
    if not data_dir:
        raise ValueError("Data directory not specified.")
    if not csv_path:
        raise ValueError("CSV file path not specified.")

    # Collect all video files
    all_files = _list_video_files_recursively(data_dir)

    # Here we filter files based on split, e.g., "train"
    split = "train"
    all_files = [f for f in all_files if f'_{split}_fixed.mp4' in f]

    print(f"Number of video files ({split}): {len(all_files)}")

    # MPI initialization
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    clip_length_in_frames = video_size[0]
    frames_between_clips = frame_gap

    meta_fname = os.path.join(
        data_dir,
        f"video_clip_f{clip_length_in_frames}_g{frames_between_clips}_r{video_fps}_{split}.pkl"
    )

    # Precompute or load precomputed metadata for VideoClips
    if not os.path.exists(meta_fname):
        if rank == 0:
            print(f"Preparing {meta_fname} ...")

        video_clips = VideoClips(
            video_paths=all_files,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            num_workers=num_workers,
            frame_rate=video_fps
        )

        # Only rank 0 writes the metadata
        if rank == 0:
            with open(meta_fname, 'wb') as f:
                pickle.dump(video_clips.metadata, f)
    else:
        if rank == 0:
            print(f"Loading {meta_fname} ...")
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
        print(f"Loaded {video_clips.num_clips()} video clips from {meta_fname}.")

    # Create the dataset
    dataset = MultimodalDataset(
        video_size=video_size,
        audio_size=audio_size,
        video_clips=video_clips,
        csv_path=csv_path,
        shard=rank,
        num_shards=size,
        random_flip=random_flip,
        audio_fps=audio_fps,
        device=device
    )

    print(f"Dataset length: {len(dataset)}")

    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return loader


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description="Multimodal Data Loader Example")
        parser.add_argument('--data_dir', type=str, default='data/videos',
                            help='Directory path for video files.')
        parser.add_argument('--csv_path', type=str, default='data/video_texts.csv',
                            help='Path to CSV file mapping video_path -> text.')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='Batch size.')
        parser.add_argument('--seconds', type=float, default=4.0,
                            help='Clip length in seconds.')
        parser.add_argument('--video_fps', type=int, default=16,
                            help='Video frames per second.')
        parser.add_argument('--audio_fps', type=int, default=16000,
                            help='Audio sampling rate.')
        parser.add_argument('--image_resolution', type=int, default=224,
                            help='Resolution for each frame (height/width).')
        parser.add_argument('--frame_gap', type=int, default=1,
                            help='Gap between frames in a clip.')
        parser.add_argument('--random_flip', action='store_true',
                            help='Use random horizontal flip for video.')
        parser.add_argument('--device', type=str, default='cpu',
                            help='Device to use (cpu or cuda).')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='num_workers')
        return parser.parse_args()

    args = parse_args()

    # Determine the number of frames in each clip
    clip_length_in_seconds = args.seconds
    video_fps = args.video_fps
    F = int(clip_length_in_seconds * video_fps)  # total frames in each clip

    # Set the desired video size [F, C, H, W]
    video_size = [
        F,                       # F
        3,                       # C
        args.image_resolution,   # H
        args.image_resolution    # W
    ]

    # Set audio spectrogram size [C, N_MELS, T]
    hop_size = 160
    T_spec = int(clip_length_in_seconds * (args.audio_fps / hop_size))
    num_mels = 256
    audio_size = [
        3,         # C (depending on how normalize_spectrogram is implemented)
        num_mels,  # N_MELS
        T_spec     # T
    ]

    # Create the DataLoader
    data_loader = load_data(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        video_size=video_size,
        audio_size=audio_size,
        frame_gap=args.frame_gap,
        random_flip=args.random_flip,
        num_workers=args.num_workers,
        video_fps=video_fps,
        audio_fps=args.audio_fps,
        device=args.device
    )

    # Example of iterating through the DataLoader
    try:
        for batch_idx, batch in enumerate(data_loader, 1):
            batch_video = batch['video']  # [N, F, C, H, W]
            batch_audio = batch['audio']  # [N, C, N_MELS, T]
            batch_text = batch['text']    # [N]

            print(f"Batch {batch_idx}:")
            print(f"  Video shape: {batch_video.shape}")
            print(f"  Audio shape: {batch_audio.shape}")
            print(f"  Text: {batch_text}")
            # Perform your training or inference here
    except KeyboardInterrupt:
        print("Data loading interrupted.")