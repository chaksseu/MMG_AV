import os
import csv
import argparse
import sys
import torch
import argparse
import pandas as pd

from ImageBind.imagebind import data
from ImageBind.imagebind.models import imagebind_model
from ImageBind.imagebind.models.imagebind_model import ModalityType

from tqdm import tqdm


def evaluate_imagebind_score(inference_save_path, device):

    # make csv for imagebind score
    base_dir = inference_save_path


    output_csv = f"{base_dir}/imagebind_score_file_pairs.csv"
    audio_dir = os.path.join(base_dir, "audio")
    video_dir = os.path.join(base_dir, "video")

    # folders = [
    #     '/workspace/MMG_Inferencce_folder/0227_avsync_audio_teacher',
    #     '/workspace/MMG_Inferencce_folder/0227_audio_teacher',
    #     '/workspace/MMG_Inferencce_folder/0227_avsync_video_teacher',
    #     '/workspace/MMG_Inferencce_folder/0227_video_teacher',
    # ]

    # audio_dir = '/home/work/kby_hgh/audio_lora_vggsound_sparse_inference_0416_1e-6/step_43800'
    # video_dir = '/home/work/kby_hgh/video_lora_vggsound_sparse_inference_0413_1e-5/step_40960'




    audio_files = set([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    if len(audio_files) == 0:
        audio_files = set([f for f in os.listdir(audio_dir) if f.endswith('.flac')])

    video_files = set([f for f in os.listdir(video_dir) if f.endswith('.mp4')])

    audio_basenames = {os.path.splitext(f)[0] for f in audio_files}
    video_basenames = {os.path.splitext(f)[0] for f in video_files}

    matching_basenames = audio_basenames.intersection(video_basenames)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for basename in sorted(matching_basenames):
            video_path = os.path.join(video_dir, basename + ".mp4") 
            audio_path = os.path.join(audio_dir, basename + ".wav")
            # audio_path = os.path.join(audio_dir, basename + ".flac")

            writer.writerow([video_path, audio_path])


    # evaluate imgagebind_score
    device = device if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)


    flist = pd.read_csv(output_csv, header=None)

    cnt = 0
    avg_score_av = 0.0

    for idx in tqdm(range(flist.shape[0]), desc="Processing files"):

        mp4_path = flist.iloc[idx][0]
        wav_path = flist.iloc[idx][1]

        if not (os.path.isfile(wav_path) and os.path.isfile(mp4_path)):
            continue

        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data([mp4_path], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([wav_path], device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        avg_score_av += cos(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]).detach().cpu().numpy()
        cnt += 1

    avg_score_av /= cnt

    return avg_score_av





def main():
    parser = argparse.ArgumentParser(description="Evaluate ImageBind Score")
    parser.add_argument(
        "--inference_save_path",
        type=str,
        default=None,
        help="audio와 video 폴더를 포함한 inference 결과 저장 경로"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="평가를 실행할 디바이스 (예: cpu 또는 cuda)"
    )
    args = parser.parse_args()

    args.inference_save_path = "/home/work/kby_hgh/0622_full_teacher_vggsound_sparse"
    args.device = "cuda:0"

    avg_score_av = evaluate_imagebind_score(args.inference_save_path, args.device)
    
    print("\n=== ImageBind Score Evaluation ===")
    print(f"Average ImageBind Score: {avg_score_av}")

if __name__ == "__main__":
    main()
