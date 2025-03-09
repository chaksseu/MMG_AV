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

def evaluate_audio_metrics(preds_folder, target_folder, metrics, clap_model, device="cpu"):
    fad_score = -1

    # Frechet Audio Distance 사용 여부
    if 'FAD' in metrics:
        frechet = FrechetAudioDistance(device=device)
        torch.manual_seed(0)

        fad_score = frechet.score(preds_folder, target_folder, limit_num=None)
        

    # Loading Clap Model
    if clap_model == 0 or clap_model == 1:
        model_clap = laion_clap.CLAP_Module(enable_fusion=False) 
    elif clap_model == 2 or clap_model == 3:
        model_clap = laion_clap.CLAP_Module(enable_fusion=True) 

    model_clap.load_ckpt(model_id=clap_model) # Download the default pretrained checkpoint.
    # Resampling rate
    new_freq = 48000



    # Get the list of filenames and set up the progress bar
    filenames = [f for f in os.listdir(preds_folder) if f.endswith('.wav') or f.endswith('.flac')]
    progress_bar = tqdm(filenames, desc='Processing')
    clap_score = []
    for filename in progress_bar:
        if filename.endswith('.wav') or filename.endswith('.flac'):
            try:
                preds_audio, _ = torchaudio.load(os.path.join(preds_folder, filename), num_frames=160000)

                if np.shape(preds_audio)[0] == 2:
                    preds_audio = preds_audio.mean(dim=0)

                clap_score.append(calculate_clap(model_clap, preds_audio, filename, new_freq))

            except Exception as e:
                print(f'Error processing {filename}: {e}')

    clap_avg = np.mean(clap_score)
    clap_std = np.std(clap_score)

    return fad_score, clap_avg, clap_std


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
        required=True,
        help="예측 오디오(.wav) 파일들이 있는 폴더 경로"
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=None,
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
        "--clap_model",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="사용할 CLAP 모델 id (0, 1, 2, 3 중 선택)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="평가를 수행할 디바이스 (예: cpu 또는 cuda)"
    )
    args = parser.parse_args()

    # # 폴더 내 파일 수 확인
    # if not check_folders(args.preds_folder, args.target_folder):
    #     print("폴더 내 파일 개수가 일치하지 않습니다. 종료합니다.")
    #     exit(1)

    # 평가 실행
    fad_score, clap_avg, clap_std = evaluate_audio_metrics(
        args.preds_folder,
        args.target_folder,
        args.metrics,
        args.clap_model,
        device=args.device
    )

    # 결과 출력
    print("\n=== 평가 결과 ===")
    if fad_score is not None:
        print(f"FAD Score: {fad_score}")
    else:
        print("FAD Score: 계산되지 않음")
    print(f"CLAP Average: {clap_avg}")
    print(f"CLAP Std: {clap_std}")

if __name__ == "__main__":
    main()
