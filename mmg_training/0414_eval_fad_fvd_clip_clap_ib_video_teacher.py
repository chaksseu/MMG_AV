# evaluate_model.py

import os
import sys
import csv
import argparse
import torch

# 프로젝트 루트 경로를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from run_audio_eval import evaluate_audio_metrics
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
            fad, clap_avg, _ = evaluate_audio_metrics(
                preds_folder=audio_inference_path,
                target_folder=audio_target_path,
                metrics=["FAD", "CLAP"],
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
                inference_save_path=args.inference_save_path,
                device=device
            )

    return fad, clap_avg, fvd, clip_avg, imagebind_score


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



    device = torch.device("cuda:6")
    print(f"[INFO] Using device: {device}")
    dataset = "vggsound_sparse" # pada70m # vggsound_sparse
    modality ="video"
    model_name_list = [0]
    checkpoint_dir_list = ["step_40960"]  # "step_0", "step_8192", "step_16384", "step_24576", "step_32768", "step_40960"
    target_path = f"/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/video"
    # /home/work/kby_hgh/MMG_panda70m_test_dataset/filtered_40f_panda70m_32s_test
    # /home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/video

    for model_name in model_name_list:
        for checkpoint_num in checkpoint_dir_list:
            inference_save_path = f"/home/work/kby_hgh/video_lora_{dataset}_inference_0413_1e-5/{checkpoint_num}"
            
            results = evaluate_model(args, device, target_path, inference_save_path, modality)

            result_csv_path = f"{modality}_teacher_eval_results.csv"
            save_results_to_csv(result_csv_path, inference_save_path, results)
            print(f"[INFO] Evaluation results saved to: {result_csv_path}")


if __name__ == "__main__":
    main()