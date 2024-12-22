# reconstruct_and_save.py

import os
import torch
import numpy as np
from torchvision import transforms as T
from moviepy.editor import ImageSequenceClip, AudioFileClip
import soundfile as sf

# multimodal_dataset.py에서 필요한 함수 및 클래스 임포트
from multimodal_dataset import load_data

# converter.py와 util.py에서 필요한 함수 및 클래스 임포트
from converter import (
    Generator,
    denormalize_spectrogram,
    MAX_WAV_VALUE
)
from utils import pad_spec, normalize, denormalize

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


# 복원된 데이터를 저장할 디렉토리 설정
OUTPUT_DIR = 'reconstructed_data_vggsound_sparse_test'
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'videos')
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'audios')
COMBINED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'combined')

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

# 텐서를 이미지 프레임으로 변환하는 함수
def tensor_to_frames(video_tensor):
    """
    비디오 텐서를 이미지 프레임 리스트로 변환합니다.
    
    :param video_tensor: [C, T, H, W] 형태의 비디오 텐서
    :return: 이미지 프레임 리스트 (H, W, C) 형태의 numpy 배열 리스트
    """
    # 텐서를 CPU로 이동하고, numpy 배열로 변환
    video_tensor = video_tensor.cpu().numpy()
    
    # 정규화 역변환: (x + 1) * 127.5
    video_tensor = (video_tensor + 1) * 127.5
    video_tensor = np.clip(video_tensor, 0, 255).astype(np.uint8)
    # 채널 순서를 변경하여 [F, H, W, C] 형태로 변환
    frames = video_tensor.transpose(0, 2, 3, 1)
    
    return frames

# 오디오 스펙트로그램을 오디오 파형으로 변환하는 함수
def spectrogram_to_waveform_vocoder(spectrogram, vocoder, device="cuda", audio_length=16000*10):
    """
    멜 스펙트로그램을 오디오 파형으로 변환합니다.
    
    :param spectrogram: [num_mels, T] 형태의 멜 스펙트로그램 (torch.Tensor)
    :param vocoder: 사전 학습된 vocoder 모델 (Generator 클래스 인스턴스)
    :param device: 사용 디바이스 ('cpu' 또는 'cuda')
    :param audio_length: 생성할 오디오의 길이 (샘플 수)
    :return: 오디오 파형 (numpy 배열)
    """
    # 스펙트로그램 역정규화
    spectrogram = denormalize(spectrogram).to(device)
    denormalized_spec = denormalize_spectrogram(spectrogram)  # [num_mels, T]
    mel_spec = denormalized_spec

    # Vocoder를 사용하여 오디오 파형 생성
    with torch.no_grad():
        waveform = vocoder.inference(mel_spec, lengths=audio_length)  # [1, T_waveform]
    
    # 텐서를 CPU로 이동하고, numpy 배열로 변환
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy().squeeze()  # [T_waveform]
    else:
        waveform = waveform.squeeze()  # 이미 numpy.ndarray인 경우
    
    return waveform

# 비디오와 오디오를 복원하여 저장하는 함수
def save_reconstructed_data(data_loader, vocoder, num_batches=10, device='cuda', seconds=4):
    """
    데이터 로더에서 배치를 가져와 비디오와 오디오를 복원하여 저장합니다.
    
    :param data_loader: 데이터 로더 객체
    :param vocoder: 사전 학습된 vocoder 모델 (Generator 클래스 인스턴스)
    :param num_batches: 저장할 배치 수
    :param device: 사용 디바이스 ('cpu' 또는 'cuda')
    """
    vocoder.to(device)
    vocoder.eval()
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        
        videos = batch['video'].to(device)  # [N, C, T, H, W]
        audios = batch['audio'].to(device)  # [N, C, N_MELS, T]
        texts = batch['text']    # [N]
        
        batch_size = videos.size(0)
        
        for i in range(batch_size):
            video_tensor = videos[i]  # [C, T, H, W]
            audio_spec = audios[i]    # [C, N_MELS, T]
            text = texts[i]
            video_filename = os.path.join(VIDEO_OUTPUT_DIR, f'batch{batch_idx}_sample{i}_{text}_video.mp4')
            audio_filename = os.path.join(AUDIO_OUTPUT_DIR, f'batch{batch_idx}_sample{i}_{text}_audio.wav')
            combined_video_filename = os.path.join(COMBINED_OUTPUT_DIR, f'batch{batch_idx}_sample{i}_{text}_combined.mp4')


            # 비디오 복원
            frames = tensor_to_frames(video_tensor)  # [F, H, W, C]
            clip = ImageSequenceClip(list(frames), fps=16)  # fps는 원본 비디오 fps로 설정
            clip.write_videofile(video_filename, codec='libx264', audio=False, verbose=False, logger=None)
            mel_spec = audio_spec

            # 오디오 파형 생성
            waveform = spectrogram_to_waveform_vocoder(mel_spec, vocoder, device=device, audio_length=int(seconds*16000))  # 예: 10초
            
            # WAV 파일로 저장
            sf.write(audio_filename, waveform, 16000)
            
            # 비디오와 오디오를 결합하여 하나의 파일로 저장 (선택 사항)
            audio_clip = AudioFileClip(audio_filename)
            video_clip = ImageSequenceClip(list(frames), fps=16)
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(combined_video_filename, codec='libx264', audio_codec='aac', verbose=False, logger=None)
            
            print(f"저장 완료: {combined_video_filename}")

# 메인 함수
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="멀티모달 데이터 복원 및 저장")
    parser.add_argument('--data_dir', type=str, default='Dataset/vggsound_sparse/vggsound_sparse', help='비디오 파일들이 저장된 데이터셋 디렉토리 경로')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--seconds', type=float, default=3.2, help='클립 길이(초)')
    parser.add_argument('--video_fps', type=int, default=12.5, help='비디오 프레임 속도')
    parser.add_argument('--audio_fps', type=int, default=16000, help='오디오 샘플링 속도')
    parser.add_argument('--image_resolution', type=int, default=256, help='이미지 해상도')
    parser.add_argument('--frame_gap', type=int, default=40, help='프레임 간격')
    parser.add_argument('--random_flip', action='store_true', help='랜덤 수평 뒤집기 사용')
    parser.add_argument('--num_batches', type=int, default=1000000, help='reconstruction해서 저장할 배치 개수')
    parser.add_argument('--num_workers', type=int, default=64, help='num_workers')
    parser.add_argument('--device', type=str, default= 'cpu', help='사용할 디바이스 (cpu 또는 cuda)')

    return parser.parse_args()

if __name__ == '__main__':

    args = main()
    
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
    
    # 데이터 로더 설정
    data_loader = load_data(
        data_dir=args.data_dir,
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
    
    # Vocoder 로드
    vocoder = Generator.from_pretrained(args.device)
    
    # 데이터 복원 및 저장
    save_reconstructed_data(data_loader, vocoder, num_batches=args.num_batches, device=args.device, seconds=args.seconds)
