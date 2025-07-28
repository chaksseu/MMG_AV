# evaluate_model.py
# evaluate_model.py
# evaluate_model.py

import os
import sys
import csv
import argparse
import torch

# 프로젝트 루트 경로를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from run_audio_eval_0416 import evaluate_audio_metrics
from run_video_eval import evaluate_video_metrics
from run_imagebind_score import evaluate_imagebind_score


def evaluate_model(args, device, target_path, inference_save_path, modality="all"):
    """
    오디오 및 비디오 결과물에 대해 FAD, CLAP, FVD, CLIP, ImageBind 점수를 평가합니다.
    """


    audio_target_path = target_path
    video_target_path = target_path

    audio_inference_path = inference_save_path
    video_inference_path = inference_save_path

    if modality == "all":
        audio_target_path = os.path.join(target_path, "audio")
        video_target_path = os.path.join(target_path, "video")

        audio_inference_path = os.path.join(inference_save_path, "audio")
        video_inference_path = os.path.join(inference_save_path, "video")

    fad, clap_avg, fvd, clip_avg, imagebind_score = 0.0, 0.0, 0.0, 0.0, 0.0


    with torch.no_grad():
        if modality == "all" or modality == "audio":
            print("[INFO] Evaluating Audio Metrics...")
            FD, FAD, KL_SIG, KL_SOFT, IS, CLAP = evaluate_audio_metrics(
                preds_folder=audio_inference_path,
                target_folder=audio_target_path,
                metrics=['CLAP', 'FAD', 'ISC', 'FD', 'KL'],
                clap_model=1,
                device=device
            )
        if modality == "all" or modality == "video":
            print("[INFO] Evaluating Video Metrics...")
            fvd, clip_avg = evaluate_video_metrics(
                preds_folder=video_inference_path,
                target_folder=video_target_path,
                metrics=["fvd", "clip"],
                device=device,
                num_frames=args.frames
            )

        if modality == "all":
            print("[INFO] Evaluating ImageBind Score...")
            imagebind_score = evaluate_imagebind_score(
                inference_save_path=inference_save_path,
                device=device
            )

    return FD, FAD, KL_SIG, KL_SOFT, IS, CLAP, fvd, clip_avg, imagebind_score


def save_results_to_csv(csv_path, inference_save_path, results):
    """
    평가 결과를 CSV 파일로 저장합니다.
    """
    headers = ["inference_save_path", "FAD", "CLAP", "FVD", "CLIP", "ImageBind"]
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([inference_save_path] + list(results))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs for audio, video, and cross-modal metrics.")
    parser.add_argument("--target_path", type=str,  default="/home/work/kby_hgh/MMG_panda70m_test_dataset/filtered_40f_panda70m_32s_test", help="GT 경로")
    parser.add_argument("--inference_save_path", type=str, default="/home/work/kby_hgh/MMG_panda70m_test_dataset/filtered_40f_panda70m_32s_test", help="모델 추론 결과 경로")
    parser.add_argument("--frames", type=int, default=40, help="FVD/CLIP 평가를 위한 프레임 수")
    parser.add_argument("--result_csv_path", type=str, default="eval_results.csv", help="결과를 저장할 CSV 파일 경로")
    parser.add_argument("--device", type=int, default=7, help="CUDA 디바이스 번호")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    device = torch.device("cuda:2")
    dataset = "vgg" # panda70m # clotho # vggsound_sparse # panda # audiocaps
    modality ="all"
    model_name_list = [0]


    checkpoint_dir_list = ["12095", "14111", "16127", "18143", "20159", "22175", "24191", "26207", "28223", "30239", "32255", "34271"]
    # checkpoint_dir_list = ["36287", "38303", "40319", "42335", "44351", "46367", "48383", "50399", "52415", "54431", "56447", "58463", "60479"]

    for model_name in model_name_list:
        for checkpoint_num in checkpoint_dir_list:
            # inference_save_path = f"/home/work/kby_hgh/MMG_Inferencce_folder/{dataset}_{model_name}_{checkpoint_num}/{modality}"

            # inference_save_path = f"/home/work/kby_hgh/MMG_Inferencce_folder/{dataset}_Original/{modality}"


            target_path = "/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320"
            # /home/work/kby_hgh/MMG_AC_test_dataset/0408_AC_test_trimmed_wavs


            # /home/work/kby_hgh/MMG_clotho_test_set/clotho_test_32s
            # /home/work/kby_hgh/MMG_panda70m_test_dataset/filtered_40f_panda70m_32s_test
            # webvid test set?
            # /home/work/kby_hgh/vggsound_sparse_test_curated_final_0320

            # inference_save_path = f"/home/work/kby_hgh/NEW_MMG_step_{checkpoint_num}_vggsound_sparse"
            # inference_save_path = f"/home/work/kby_hgh/MMG_NAIVE_DISTILL_step_{checkpoint_num}_vggsound_sparse"
            # inference_save_path = f"/home/work/kby_hgh/MMG_ONLY_RC_DISTILL_{dataset}_step_{checkpoint_num}"
            # inference_save_path = f"/home/work/kby_hgh/MMG_FILTERED_ONLY_RC_DISTILL_{dataset}_step_{checkpoint_num}"

            # inference_save_path = f"/home/work/kby_hgh/FINAL_ALL_MMG_OURS_step_{checkpoint_num}_vggsound_sparse"
            # inference_save_path = f"/home/work/kby_hgh/FINAL_FILTERED_MMG_OURS_step_{checkpoint_num}_vggsound_sparse"
            # inference_save_path = f"/home/work/kby_hgh/MMG_FINAL_NAIVE_DISTILL_step_{checkpoint_num}_vggsound_sparse"
            # inference_save_path = f"/home/work/kby_hgh/MMG_FINAL_FILTERED_RC_DISTILL_vgg_step_{checkpoint_num}"

            # inference_save_path = f"/home/work/kby_hgh/MMG_FINAL_FILTERED_RC_DISTILL_LINEAR_T_vgg_step_{checkpoint_num}"
            inference_save_path = f"/home/work/kby_hgh/MMG_FINAL_FILTERED_RC_DISTILL_LINEAR_T_1e5_vgg_step_{checkpoint_num}"


            results = evaluate_model(args, device, target_path, inference_save_path, modality)

            result_csv_path = f"MMG_final_{dataset}_eval_results.csv"
            save_results_to_csv(result_csv_path, inference_save_path, results)
            print(f"[INFO] Evaluation results saved to: {result_csv_path}")


if __name__ == "__main__":
    main()


# /home/work/kby_hgh/MMG_Inferencce_folder/tensorboard/0413_MMG_OURS_1e-4_8gpu_videocaption_continue_step_10175_panda/video