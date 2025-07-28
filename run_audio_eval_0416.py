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

from torch.utils.data import DataLoader
from utils.load_mel import WaveDataset
from audio_metrics.clap_score import calculate_clap
from audio_metrics.fad import FrechetAudioDistance
from audio_metrics.isc import calculate_isc
from audio_metrics.kl import calculate_kl
from feature_extractors.panns import Cnn14
from audio_metrics.fid import calculate_fid


def check_folders(preds_folder, target_folder):
    preds_files = [f for f in os.listdir(preds_folder) if f.endswith('.wav')]
    target_files = [f for f in os.listdir(target_folder) if f.endswith('.wav')]
    if len(preds_files) != len(target_files):
        print('Mismatch in number of files between preds and target folders.')
        return False
    return True

def get_featuresdict( dataloader, device, mel_model):
    out = None
    out_meta = None

    for waveform, filename in tqdm(dataloader):
        
        metadict = {
            'file_path_': filename,
        }
        waveform = waveform.squeeze(1)

        waveform = waveform.float().to(device)

        with torch.no_grad():
            featuresdict = mel_model(waveform) # 'logits': [1, 527]

        featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

        if out is None:
            out = featuresdict
        else:
            out = {k: out[k] + featuresdict[k] for k in out.keys()}

        if out_meta is None:
            out_meta = metadict
        else:
            out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    return {**out, **out_meta}


def evaluate_audio_metrics(preds_folder, target_folder, metrics, clap_model, device="cpu"):
    fad_score = -1

    out = {}


    if 'FAD' in metrics or 'KL' in metrics or 'ISC' in metrics or 'FD' in metrics:

        backbone = 'cnn14'
        sampling_rate = 16000
        frechet = FrechetAudioDistance(device=device)
        
        frechet.model = frechet.model.to(device)

        if sampling_rate == 16000:
            mel_model = Cnn14(
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
        else:
            raise ValueError(
                'We only support the evaluation on 16kHz sampling rate.'
            )

        mel_model.eval()
        mel_model.to(device)
        fbin_mean, fbin_std = None, None

    
        torch.manual_seed(0)

        num_workers = 4

        outputloader = DataLoader(
            WaveDataset(
                preds_folder,
                sampling_rate, 
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )

        resultloader = DataLoader(
            WaveDataset(
                target_folder,
                sampling_rate, 
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )



        print('FAD - Extracting features from %s.' % target_folder)
        featuresdict_2 = get_featuresdict(resultloader, device, mel_model)
        
        print('FAD - Extracting features from %s.' % preds_folder)
        featuresdict_1 = get_featuresdict(outputloader, device, mel_model)

        # FAD
        if 'FAD' in metrics:
            fad_score = frechet.score(preds_folder, target_folder, limit_num=None)
            out['frechet_audio_distance'] = fad_score
        
        if check_folders(preds_folder, target_folder) and 'KL' in metrics:
            kl_sigmoid, kl_softmax, kl_ref, paths_1 = calculate_kl(
                featuresdict_1, featuresdict_2, 'logits', True
            )
            out['kullback_leibler_divergence_sigmoid'] = float(kl_sigmoid)
            out['kullback_leibler_divergence_softmax'] =  float(kl_softmax)


        if 'ISC' in metrics:
            print('ISC - Extracting features from %s.' % preds_folder)
            # featuresdict_1 = get_featuresdict(outputloader, device, mel_model)

            mean_isc, std_isc = calculate_isc(
                featuresdict_1,
                feat_layer_name='logits',
                splits=10,
                samples_shuffle=True,
                rng_seed=2020,
            )
            out['inception_score_mean'] =  mean_isc
            out['inception_score_std'] = std_isc


        if 'FD' in metrics:
            print('FD - Extracting features from %s.' % target_folder)
            featuresdict_2 = get_featuresdict(resultloader, device, mel_model)
            
            print('FD - Extracting features from %s.' % preds_folder)
            # featuresdict_1 = get_featuresdict(outputloader, device, mel_model)

            if('2048' in featuresdict_1.keys() and '2048' in featuresdict_2.keys()):
                metric_fid = calculate_fid(
                    featuresdict_1, featuresdict_2, feat_layer_name='2048'
                )
                out['frechet_distance'] = round(metric_fid, 3)


    # Loading Clap Model # we use 1
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
                # target_audio, _ = torchaudio.load(os.path.join(target_folder, filename), num_frames=160000)
                # min_len = min(preds_audio.size(1), target_audio.size(1))
                # preds_audio, target_audio = preds_audio[:, :min_len], target_audio[:, :min_len]

                # if np.shape(target_audio)[0] == 2:
                #     target_audio = target_audio.mean(dim=0)
                if np.shape(preds_audio)[0] == 2:
                    preds_audio = preds_audio.mean(dim=0)


                clap_score.append(calculate_clap(model_clap, preds_audio, filename, new_freq))

            except Exception as e:
                print(f'Error processing {filename}: {e}')

    clap_avg = np.mean(clap_score)
    out['clap'] = np.mean(clap_score)
    clap_std = np.std(clap_score)

    return out['frechet_distance'], out['frechet_audio_distance'], out['kullback_leibler_divergence_sigmoid'], out['kullback_leibler_divergence_softmax'], out['inception_score_mean'], out['clap']


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
        default="/home/work/kby_hgh/audio_lora_vggsound_sparse_inference_0416_1e-6/step_0",
        help="예측 오디오(.wav) 파일들이 있는 폴더 경로"
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/audio",
        help="타겟 오디오(.wav) 파일들이 있는 폴더 경로"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=['CLAP', 'FAD', 'ISC', 'FD', 'KL'],
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
        default="cuda:0",
        help="평가를 수행할 디바이스 (예: cpu 또는 cuda)"
    )
    args = parser.parse_args()

    # # 폴더 내 파일 수 확인
    # if not check_folders(args.preds_folder, args.target_folder):
    #     print("폴더 내 파일 개수가 일치하지 않습니다. 종료합니다.")
    #     exit(1)

    if args.target_folder == None or not check_folders(args.preds_folder, args.target_folder):
        text = 'Running only reference-free metrics'
        same_name = False



    # 평가 실행
    FD, FAD, KL_SIG, KL_SOFT, IS, CLAP = evaluate_audio_metrics(
        args.preds_folder,
        args.target_folder,
        args.metrics,
        args.clap_model,
        device=args.device
    )

    # 결과 출력
    print("\n=== 평가 결과 ===")
    print(f"FD: {FD}")
    print(f"FAD: {FAD}")
    print(f"KL_SIG: {KL_SIG}")
    print(f"KL_SOFT: {KL_SOFT}")
    print(f"IS: {IS}")
    print(f"CLAP: {CLAP}")


if __name__ == "__main__":
    main()
