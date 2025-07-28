import os
import sys
import csv
import argparse
import torch

# 프로젝트 루트 경로를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from run_video_eval_0603 import evaluate_video_metrics


def evaluate_model(args, device, target_path, inference_save_path, modality, csv_path):
    """
    오디오 및 비디오 결과물에 대해 FAD, CLAP, FVD, CLIP, ImageBind 점수를 평가합니다.
    """

    video_target_path = target_path

    video_inference_path = inference_save_path


    fvd, clip_avg = 0.0, 0.0


    with torch.no_grad():
        if modality == "all" or modality == "video":
            print("[INFO] Evaluating Video Metrics...")
            fvd, clip_avg = evaluate_video_metrics(
                preds_folder=video_inference_path,
                target_folder=video_target_path,
                metrics=["fvd", "clip"],
                device=device,
                num_frames=args.frames,
                csv_path=csv_path
            )

    return fvd, clip_avg


def save_results_to_csv(csv_path, inference_save_path, results):
    """
    평가 결과를 CSV 파일로 저장합니다.
    """
    headers = ["inference_save_path","FVD", "CLIP"]
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
    dataset = "openvid" # vggsound_sparse # openvid
    modality ="video"
    model_name_list = ['LORA'] # 'FULL', 'LORA'


    # checkpoint_dir_list = ["0", "16384"]
    # checkpoint_dir_list = ["32768", "49152"]
    # checkpoint_dir_list = ["65536", "81920"]
    # checkpoint_dir_list = ["98304", "114688"]
    # checkpoint_dir_list = ["131072", "147456"]

    checkpoint_dir_list = ["65536"]

    for model_name in model_name_list:
        for checkpoint_num in checkpoint_dir_list:
            
            if dataset == "vggsound_sparse":
                target_path = "/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/video"
                csv_path = "/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/vggsound_sparse_curated_292.csv"
            if dataset == "openvid":
                target_path = "/home/work/kby_hgh/MMG_01/video_lora_training/processed_OpenVid_2000_test_videos_42"
                csv_path = "/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_test_with_audio_caption.csv"

            inference_save_path = f"/home/work/kby_hgh/0603_video_teacher_{model_name}_{dataset}_inference/checkpoint-step-{checkpoint_num}"

            results = evaluate_model(args, device, target_path, inference_save_path, modality, csv_path)

            result_csv_path = f"0605_video_teahcer_{dataset}_eval_results.csv"
            save_results_to_csv(result_csv_path, inference_save_path, results)
            print(f"[INFO] Evaluation results saved to: {result_csv_path}")


if __name__ == "__main__":
    main()


# /home/work/kby_hgh/MMG_Inferencce_folder/tensorboard/0413_MMG_OURS_1e-4_8gpu_videocaption_continue_step_10175_panda/video