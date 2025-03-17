import os
import argparse
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
import multiprocessing
from functools import partial

# 필요하다면 sys.path 설정
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# converter_copy_0123.py 안의 함수들을 사용한다고 가정
from preprocess.converter_copy_0123 import (
    get_mel_spectrogram_from_audio,
    normalize_spectrogram,
)

from preprocess.utils import pad_spec



def process_audio(audio_id, args):
    """
    단일 오디오 파일을 처리하여 Mel Spectrogram을 생성하고 저장합니다.
    """
    spec_save_path = os.path.join(args.spectrogram_dir, f"{audio_id}.pt")
    if os.path.exists(spec_save_path):
        # 이미 생성된 경우 스킵
        return

    # audio_path = os.path.join(args.audio_dir, f"{audio_id}.flac")


    audio_path_flac = os.path.join(args.audio_dir, f"{audio_id}.flac")
    audio_path_mp3 = os.path.join(args.audio_dir, f"{audio_id}.wav")

    # flac이 있으면 사용, 없으면 mp3 사용
    audio_path = audio_path_flac if os.path.exists(audio_path_flac) else audio_path_mp3


    if not os.path.isfile(audio_path):
        # 혹은 .wav, .mp3 등 확장자 다양하다면 csv에서 확장자도 읽어야 함
        print(f"[Warning] Audio file not found: {audio_path}")
        return

    try:
        # 오디오 로딩
        waveform, sr = torchaudio.load(audio_path)
        # 리샘플
        if sr != args.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=args.sample_rate)
            waveform = resampler(waveform)

        # 스테레오 -> 모노
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # CPU 상에서 Mel Spectrogram 계산
        audio_np = waveform.squeeze(0).cpu().numpy()
        _, spec = get_mel_spectrogram_from_audio(audio_np)
        
        #spec = normalize_spectrogram(spec)  # (n_mels, T)


        #print(spec.shape)
        # .pt로 저장
        torch.save(spec, spec_save_path)
    except Exception as e:
        print(f"[Error] Processing {audio_id} failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='/workspace/vggsound_processing/New_VGGSound.csv', help="CSV 파일 경로")
    parser.add_argument("--audio_dir", type=str, default="/workspace/data/preprocessed_VGGSound_train_audio_0310", help="오디오(flac) 파일 폴더 경로")
    parser.add_argument("--spectrogram_dir", type=str, default="/workspace/data/preprocessed_VGGSound_train_spec_0310", help="미리 생성된 Mel Spectrog")# /home/rtrt5060/preprocessed_spec, /home/jupyter/MMG_TA_dataset_audiocaps_wavcaps/preprocessed_spec
    parser.add_argument("--sample_rate", type=int, default=16000, help="오디오 리샘플링 레이트")
    parser.add_argument("--n_mels", type=int, default=256, help="Mel bands 개수")
    parser.add_argument("--hop_size", type=int, default=160, help="hop size")
    parser.add_argument("--num_workers", type=int, default=8, help="병렬 처리에 사용할 CPU 코어 수")
    args = parser.parse_args()

    os.makedirs(args.spectrogram_dir, exist_ok=True)

    # CSV 읽기
    df = pd.read_csv(args.csv_path)

    # 'train' 데이터만 필터링
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        if train_df.empty:
            print("[Warning] 'train' 데이터가 존재하지 않습니다.")
            return
    else:
        print("[Warning] 'split' 컬럼이 존재하지 않습니다. 'train' 데이터로 간주할 수 있는 다른 기준을 사용하세요.")
        return

    # 중복 id(오디오 파일 이름) 제거 (train 데이터만)
    unique_audio_ids = train_df["id"].unique()

    # 병렬 처리 함수 설정
    process_func = partial(process_audio, args=args)

    # 멀티프로세싱 풀 생성
    with multiprocessing.Pool(processes=4) as pool:
        # tqdm을 사용하여 진행 상황 표시
        list(tqdm(pool.imap_unordered(process_func, unique_audio_ids), total=len(unique_audio_ids), desc="Precomputing spectrograms"))

    print("모든 'train' Mel Spectrogram이 사전 계산되어 저장되었습니다.")

if __name__ == "__main__":
    main()
