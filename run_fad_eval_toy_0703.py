'''
Audio and Visual Evaluation Toolkit

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT		

Description:
Audio Evaluation - run_audio_eval.py
This toolbox includes the following metrics:
- FAD: Frechet audio distance
- ISc: Inception score
- FD: Frechet distance, realized by PANNs, a state-of-the-art audio classification model
- KL: KL divergence (softmax over logits)
- KL_Sigmoid: KL divergence (sigmoid over logits)
- SI_SDR: Scale-Invariant Signal-to-Distortion Ratio
- SDR: Signal-to-Distortion Ratio
- SI_SNR: Scale-Invariant Signal-to-Noise Ratio
- SNR: Signal-to-Noise Ratio
- PESQ: Perceptual Evaluation of Speech Quality
- STOI: Short-Time Objective Intelligibility
- CLAP-Score: Implemented with LAION-AI/CLAP

### Running the metris
python run_audio_eval.py --preds_folder /path/to/generated/audios --target_folder /path/to/the/target_audios \
--metrics SI_SDR SDR SI_SNR SNR PESQ STOI CLAP FAD ISC FD KL --results NAME_YOUR_RESULTS_FILE.txt


Third-Party Snippets/Credits:

[1] - Taken from [https://github.com/haoheliu/audioldm_eval] - [MIT License]
    - Adapted code for FAD, ISC, FID, and KL computation

[2] - Taken from [https://github.com/LAION-AI/CLAP] - [CC0-1.0 license]
    - Snipped utilized for audio embeddings and text embeddings retrieval

'''
import argparse
import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import laion_clap
from audio_metrics.clap_score import calculate_clap
from audio_metrics.fad import FrechetAudioDistance



def check_folders(preds_folder, target_folder):
    preds_files = [f for f in os.listdir(preds_folder) if f.endswith('.wav')]
    target_files = [f for f in os.listdir(target_folder) if f.endswith('.wav')]
    if len(preds_files) != len(target_files):
        print('Mismatch in number of files between preds and target folders.')
        return False
    return True

def evaluate_audio_metrics(preds_folder, target_folder, metrics, device="cpu"):
    fad_score = -1

    # Frechet Audio Distance 사용 여부
    if 'FAD' in metrics:
        frechet = FrechetAudioDistance(device=device)
        torch.manual_seed(0)

        fad_score = frechet.score(preds_folder, target_folder, limit_num=None)
        
    return fad_score


# Defining clap model descriptions
CLAP_MODEL_DESCRIPTIONS = {
    0: '630k non-fusion ckpt',
    1: '630k+audioset non-fusion ckpt',
    2: '630k fusion ckpt',
    3: '630k+audioset fusion ckpt'
}





def main():

    parser = argparse.ArgumentParser(description="Evaluate Audio Metrics")
    parser.add_argument(
        "--preds_folder",
        type=str,
        default="True",
        help="예측 오디오(.wav) 파일들이 있는 폴더 경로"
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default="True",
        help="타겟 오디오(.wav) 파일들이 있는 폴더 경로"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=["FAD"],
        help="평가할 메트릭 리스트 (예: FAD)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="평가를 수행할 디바이스 (예: cpu 또는 cuda)"
    )
    args = parser.parse_args()

    # 평가 실행
    args.preds_folder = "/home/work/kby_hgh/MMG_01/toy_mmg/tts_digit_wav/test_all"
    args.target_folder = "/home/work/kby_hgh/MMG_01/toy_mmg/tts_digit_wav/test_all"

    fad_score = evaluate_audio_metrics(
        args.preds_folder,
        args.target_folder,
        args.metrics,
        device=args.device
    )

    # 결과 출력
    print("\n=== 평가 결과 ===")
    if fad_score is not None:
        print(f"FAD Score: {fad_score}")
    else:
        print("FAD Score: 계산되지 않음")


if __name__ == "__main__":
    main()
