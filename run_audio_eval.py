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

    # Frechet Audio Distance 사용 여부
    if 'FAD' in metrics:
        frechet = FrechetAudioDistance(device=device)
        torch.manual_seed(0)

        # FAD 계산
        if 'FAD' in metrics:
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
    filenames = [f for f in os.listdir(preds_folder) if f.endswith('.wav')]
    progress_bar = tqdm(filenames, desc='Processing')
    clap_score = []
    for filename in progress_bar:
        if filename.endswith('.wav'):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate audio on acoustic metrics.')

    parser.add_argument('--preds_folder', required=True, help='Path to the folder with predicted audio files.')
    parser.add_argument('--target_folder', required=False, default=None, help='Path to the folder with target audio files.')
    
    parser.add_argument('--metrics', nargs='+',
                        choices=['SI_SDR', 'SDR', 'SI_SNR', 'SNR', 'PESQ', 'STOI', 'CLAP', 'FAD', 'ISC', 'FD', 'KL'],
                        help='List of metrics to calculate.')
    
    parser.add_argument('--clap_model', type=int, default=1, help='CLAP model id for score computations.')
    parser.add_argument('--results_file', required=True, help='Path to the text file to save the results.')
    
    # device 인자 추가 (기본값 "cpu")
    parser.add_argument('--device', default="cpu", help='Device to use: "cpu" or "cuda"')

                        
    args = parser.parse_args()
    evaluate_audio_metrics(
        args.preds_folder,
        args.target_folder,
        args.metrics,
        args.results_file,
        args.clap_model,
        device=args.device
    )