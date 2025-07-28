import os
import sys
import csv
import argparse
import torch

# 프로젝트 루트 경로를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# from run_audio_eval_0416 import evaluate_audio_metrics
from run_audio_eval_0611 import evaluate_audio_metrics
from run_video_eval_0603 import evaluate_video_metrics

from run_imagebind_score import evaluate_imagebind_score


def evaluate_model(args, device, target_path, inference_save_path, csv_path, modality="all"):
    """
    오디오 및 비디오 결과물에 대해 FAD, CLAP, FVD, CLIP, ImageBind 점수를 평가합니다.
    """


    # audio_target_path = os.path.join(target_path, "audio")
    # video_target_path = os.path.join(target_path, "video")

    audio_target_path = target_path

    audio_inference_path = os.path.join(inference_save_path, "audio")
    video_inference_path = os.path.join(inference_save_path, "video")

    # audio_inference_path = inference_save_path

    # if modality == "all":
    #     audio_target_path = os.path.join(target_path, "audio")
    #     video_target_path = os.path.join(target_path, "video")

    #     audio_inference_path = os.path.join(inference_save_path, "audio")
    #     video_inference_path = os.path.join(inference_save_path, "video")



    FD, FAD, KL_SOFT, IS, CLAP, fvd, clip_avg, imagebind_score = 0.0, 0.0, 0.0, 0.0, 0.0, {'final': 0.0}, 0.0, [0.0]


    with torch.no_grad():
        if modality == "all" or modality == "audio":
            print("[INFO] Evaluating Audio Metrics...")
            FD, FAD, KL_SIG, KL_SOFT, IS, CLAP = evaluate_audio_metrics(
                preds_folder=audio_inference_path,
                target_folder=audio_target_path,
                metrics=['CLAP', 'FAD', 'ISC', 'FD', 'KL'],
                clap_model=1,
                device=device,
                csv_path=csv_path
            )
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

        # print("[INFO] Evaluating ImageBind Score...")
        # imagebind_score = evaluate_imagebind_score(
        #     inference_save_path=inference_save_path,
        #     device=device
        # )

    # return FD, FAD, KL_SOFT, IS, CLAP, fvd['final'], clip_avg, imagebind_score[0]
    return FD, FAD, KL_SOFT, IS, CLAP


def save_results_to_csv(csv_path, inference_save_path, results):
    """
    평가 결과를 CSV 파일로 저장합니다.
    """
    headers = ["inference_save_path", "FD", "FAD", "KL", "IS", "CLAP"]
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


    device = torch.device("cuda:4")
    dataset = "audiocaps" # vggsound_sparse # audiocaps # openvid
    modality ="audio"


    # model_name_list = ["/home/work/kby_hgh/0714_ALL_FULL_MMG_OURS_step"]
    # model_name_list = ["/home/work/kby_hgh/0710_FILTERED_FULL_MMG_OURS_step"]
    # model_name_list = ["/home/work/kby_hgh/0714_FILTERED_FULL_MMG_NAIVE_DISTILL_step"]
    # model_name_list = ["/home/work/kby_hgh/0710_FILTERED_FULL_MMG_RC_LENEAR_T_1_2_1_2_step"]

    # checkpoint_dir_list = ["4031", "8063", "12095", "16127", "20159", "24191", "28223"] # filtered ours, naive
    checkpoint_dir_list = ["2015", "4031", "6047", "8063", "10079", "12095", "14111", "16127", "18143", "20159", "22175", "24191", "26207", "28223"]


    model_name_list = ["/home/work/kby_hgh/0717_ALL_FULL_MMG_OURS_step"]
    # model_name_list = ["/home/work/kby_hgh/0717_FILTERED_FULL_MMG_NAIVE_DISTILL_step"]


    checkpoint_dir_list = ["2015", "4031", "6047", "8063", "10079", "12095", "14111", "16127", "18143", "20159", "22175", "24191", "26207", "28223"]

    for model_name in model_name_list:
        for checkpoint_num in checkpoint_dir_list:

            # VGG_CSV_PATH="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/vggsound_sparse_curated_292.csv"
            # VGG_GT_TEST_PATH="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320"

            # OPENVID_CSV_PATH="/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_test_with_audio_caption.csv"     
            # OPENVID_GT_TEST_PATH="/home/work/kby_hgh/MMG_01/video_lora_training/processed_OpenVid_2000_test_videos_42"

            # AC_CSV_PATH="/home/work/kby_hgh/MMG_AC_test_dataset/0407_one_cap_AC_test_with_video_caption.csv"
            # AC_GT_TEST_PATH="/home/work/kby_hgh/MMG_AC_test_dataset/0408_AC_test_trimmed_wavs"


            target_path = "/home/work/kby_hgh/MMG_AC_test_dataset/0627_AC_test_split_wavs"
            csv_path = "/home/work/kby_hgh/MMG_AC_test_dataset/0627_one_cap_AC_split_test_with_video_caption.csv"
            
            # inference_save_path = f"{model_name}/{checkpoint_num}"
            inference_save_path = f"{model_name}_{checkpoint_num}_{dataset}"

            if not os.path.isdir(inference_save_path):
                print(f"{inference_save_path} does not exist. Skipping")
                continue

            results = evaluate_model(args, device, target_path, inference_save_path, csv_path, modality)

            result_csv_path = f"{model_name}_{dataset}_eval_results.csv"
            save_results_to_csv(result_csv_path, inference_save_path, results)
            print(f"[INFO] Evaluation results saved to: {result_csv_path}")


if __name__ == "__main__":
    main()