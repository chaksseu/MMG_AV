import os
import torch
import torch.nn.functional as F
from multimodal_dataset import load_data
import csv
from pathlib import Path


# 데이터를 저장할 기본 폴더 설정
OUTPUT_DIR = "_processed_data_vggsound_sparse_32s_40frame_new_noramlization"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
TEXT_DIR = os.path.join(OUTPUT_DIR, "text")
CSV_FILE = os.path.join(OUTPUT_DIR, "dataset_info.csv")


# 폴더 생성 함수
def create_output_directory():
    """
    출력 폴더를 생성합니다.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(VIDEO_DIR).mkdir(parents=True, exist_ok=True)
    Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)

# CSV 파일 생성 함수
def create_csv_file():
    """
    CSV 파일을 생성합니다.
    """
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Video", "Audio", "Text"])
        print(f"CSV 파일 생성: {CSV_FILE}")

def save_to_csv(Video, Audio, text):
    """
    파일 이름과 텍스트 정보를 CSV 파일에 저장합니다.
    """
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([Video, Audio, text])
    print(f"CSV에 데이터 저장 - Video: {Video}, Audio: {Audio}, Text: {text}")


def save_item(video: torch.Tensor, audio: torch.Tensor, index: int):
    """
    전처리된 데이터를 디스크에 저장하고 CSV 파일에 파일 이름과 텍스트 정보를 기록합니다.
    """
    # 비디오 저장
    video_save_path = os.path.join(VIDEO_DIR, f"video_{index}.pt")
    torch.save(video, video_save_path)

    # 오디오 저장
    audio_save_path = os.path.join(AUDIO_DIR, f"audio_{index}.pt")
    torch.save(audio, audio_save_path)

    print(f"데이터 저장 완료 - 비디오: {video_save_path}, 오디오: {audio_save_path}")


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description="멀티모달 데이터 로더 예제")
        parser.add_argument('--data_dir', type=str, default='Dataset/vggsound_sparse/vggsound_sparse', help='비디오 파일들이 저장된 데이터셋 디렉토리 경로')
        parser.add_argument('--batch_size', type=int, default=1, help='배치 크기')
        parser.add_argument('--seconds', type=float, default=3.2, help='클립 길이(초)')
        parser.add_argument('--video_fps', type=float, default=12.5, help='비디오 프레임 속도')
        parser.add_argument('--audio_fps', type=int, default=16000, help='오디오 샘플링 속도')
        parser.add_argument('--image_resolution', type=int, default=256, help='이미지 해상도')
        parser.add_argument('--frame_gap', type=int, default=40, help='프레임 간격')
        parser.add_argument('--num_workers', type=int, default=64, help='num_workers')
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
        num_workers=args.num_workers,
        video_fps=video_fps,
        audio_fps=args.audio_fps,
        device=args.device
    )


    


    # 출력 폴더 및 CSV 파일 생성
    create_output_directory()
    create_csv_file()

    # 이후 데이터 처리 로직
    print("args.batch_size", args.batch_size)

    try:
        for batch_idx, batch in enumerate(data_loader):
            batch_video = batch['video']  # [N, F, C, H, W] [B, 64, 3, 256, 256]
            batch_audio = batch['audio']  # [N, C, H, W] [B, 3, 256, 400]
            batch_text = batch['text']    # [N]
            batch_video = batch_video.detach().cpu()
            batch_audio = batch_audio.detach().cpu()

            
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
            


            # 각 데이터 항목 저장 및 CSV 기록
            for i in range(len(batch_video)):
                index = batch_idx * args.batch_size + i
                #print(f"batch_video[{i}]", batch_video[i].shape)
                #print(f"batch_audio[{i}]", batch_audio[i].shape)

                save_item(batch_video[i], batch_audio[i], index)
                video_file_name = f"video_{index}.pt"
                audio_file_name = f"audio_{index}.pt"
                save_to_csv(video_file_name, audio_file_name, batch_text[i])
            print(f"배치 {batch_idx} 처리 완료.")
    except KeyboardInterrupt:
        print("데이터 로딩 중단.")