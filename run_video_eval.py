'''
Audio and Visual Evaluation Toolkit

Author: Lucas Goncalves
Date Created: 2023-08-16 16:34:44 PDT
Last Modified: 2023-08-24 9:27:30 PDT		

Description:
Video Evaluation - run_video_eval.py
This toolbox includes the following metrics:
- FVD: Frechet Video Distance
- FID: Frechet Inception distance, realized by inceptionv3
- KID: Kernel Inception Distance
- LPIPS: Learned Perceptual Image Patch Similarity
- MiFID: Memorization-Informed Frechet Inception Distance
- SSIM: Structural Similarity Index Measure
- MS-SSIM: Multi-Scale SSIM
- PSNR: Peak Signal-to-Noise Ratio
- PSNRB: Peak Signal To Noise Ratio With Blocked Effect
- VMAF: Video Multi-Method Assessment Fusion
- VIF: Visual Information Fidelity
- CLIP-Score: Implemented with CLIP VIT model

### Running the metrics
python3 run_video_eval.py --preds_folder /path/to/generated/videos --target_folder /path/to/the/target/videos \
--num_frames {Number of frames in your video or to be used for evaluation} --output path/to/NAME_YOUR_RESULTS_FILE.txt


'''
import os
import cv2
import torch
import torchvision.transforms as transforms
import argparse
from visual_metrics.calculate_fvd import calculate_fvd
from visual_metrics.calculate_fid import calculate_fid
from visual_metrics.calculate_kid import calculate_kid
from visual_metrics.calculate_psnr import calculate_psnr
from visual_metrics.calculate_psnrb import calculate_psnrb
from visual_metrics.calculate_ssim import calculate_ssim
from visual_metrics.calculate_lpips import calculate_lpips
from visual_metrics.calculate_ms_ssim import calculate_ms_ssim
from visual_metrics.calculate_clip import calculate_clip
from visual_metrics.calculate_mifid import calculate_mifid
from visual_metrics.calculate_vmaf import calculate_vmaf
from visual_metrics.calculate_vif import calculate_vif
import json

import re  
import pandas as pd


def load_video_frames(video_path, num_frames, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for k in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((frame_size[0], frame_size[1])),
            transforms.CenterCrop((frame_size[0], frame_size[1])),
            transforms.ToTensor()
        ])
        tensor_frame = transform(frame)        
        frames.append(tensor_frame)
    cap.release()
    return torch.stack(frames)
    

def load_videos_from_folder(folder_path, num_frames):
    videos_tensor_list_orig = []
    vid_fnames = sorted(os.listdir(folder_path))
    for video_name in vid_fnames:
        video_path = os.path.join(folder_path, video_name)
        video_tensor = load_video_frames(video_path, num_frames)
        videos_tensor_list_orig.append(video_tensor)
    return torch.stack(videos_tensor_list_orig)


excluded_words = ['clip', 'test', 'sparse', 'vggsound','batch', 'proc', 'sample', 'audio', 'video']
pattern_words = re.compile(r'^(?:' + '|'.join(excluded_words) + r')\d*$', re.IGNORECASE)
pattern_numbers = re.compile(r'^\d+$')

def clean_sentence(filename):
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    
    sentence = filename.replace('_', ' ').replace('.wav', '')
    words = sentence.split()
    filtered_words = [
        word for word in words 
        if not pattern_words.match(word) and not pattern_numbers.match(word)
    ]
    cleaned_sentence = ' '.join(filtered_words)
    return cleaned_sentence

def get_new_caption_from_csv(cleaned, csv_path="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/vggsound_sparse_curated_292.csv"):
    # cleaned = clean_sentence(filename)
    
    # CSV 읽기
    df = pd.read_csv(csv_path)
    
    # caption 열에서 cleaned_sentence와 일치하는 행 찾기
    match = df[df['caption'] == cleaned]
    
    if not match.empty:
        return match.iloc[0]['new_caption']
    else:
        return None  # 또는 'Not found' 등 사용자 정의 처리

def get_new_caption_from_csv_0603(id, csv_path="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/vggsound_sparse_curated_292.csv"):
    
    # CSV 읽기
    df = pd.read_csv(csv_path)
    
    # caption 열에서 cleaned_sentence와 일치하는 행 찾기
    match = df[df['caption'] == id]
    
    if not match.empty:
        return match.iloc[0]['new_caption']
    else:
        return None  # 또는 'Not found' 등 사용자 정의 처리

def load_videos_with_caps(folder_path, num_frames):
    videos_tensor_list_orig = []
    caps = []
    # open and read the file
    vid_fnames = sorted(os.listdir(folder_path))
    for video_name in vid_fnames:
        fname = video_name[:-4]
        video_caption = get_new_caption_from_csv(clean_sentence(fname))
        print(video_caption)
        caps.append(video_caption)
        video_path = os.path.join(folder_path, video_name)
        video_tensor = load_video_frames(video_path, num_frames)
        videos_tensor_list_orig.append(video_tensor)
    
    # print(caps)
    
    return torch.stack(videos_tensor_list_orig), caps

def process_fvd(data):
    fvd_final = data["fvd"]["fvd"]["final"]

    result = []
    try:
        fvd_16 = data["fvd"]["fvd"]["[:16]"]
        fvd_24 = data["fvd"]["fvd"]["[:24]"]
        result.append(f'FVD_16: {fvd_16:.3f}\n')
        result.append(f'FVD_24: {fvd_24:.3f}\n')
        result.append(f'FVD: {fvd_final:.3f}\n')
    except:
        result.append(f'FVD: {fvd_final:.3f}\n')
    return ''.join(result)

def process_metric(metric_name, data):
    result = []
    for key, value in data[metric_name][metric_name].items():
        if 'avg' in key:
            avg_key = key
            std_key = key.replace("avg", "std")
            num = avg_key.split('[:')[1].split(']')[0]
            avg_value = value
            std_value = data[metric_name][f"{metric_name}_std"][std_key]
            result.append(f'{metric_name.upper()}_{num}: Average = {avg_value:.3f}, Std = {std_value:.3f}\n')
    final_avg = data[metric_name][metric_name]["final"]
    final_std = data[metric_name][f"{metric_name}_std"]["final"]
    result.append(f'{metric_name.upper()}: Average = {final_avg:.3f}, Std = {final_std:.3f}\n')

    return ''.join(result)



def evaluate_video_metrics(preds_folder, target_folder, metrics, device, num_frames):

    fvd_score = {}

    fvd_score["fvd"] = -1

    calculate_final = True
    calculate_per_frame = 8

    if 'fvd' in metrics:
        orig_videos = load_videos_from_folder(target_folder, num_frames)
        new_videos = load_videos_from_folder(preds_folder, num_frames)
        fvd_score = calculate_fvd(orig_videos, new_videos, calculate_per_frame, calculate_final, device)

    if 'clip' in metrics:
        clip_videos, caps = load_videos_with_caps(preds_folder, num_frames)
        clip_score = calculate_clip(clip_videos, caps, calculate_per_frame, calculate_final, 'openai/clip-vit-base-patch16', device)

    return fvd_score["fvd"], clip_score["clip"]['final']





def main():
    parser = argparse.ArgumentParser(description="Evaluate Video Metrics")
    parser.add_argument(
        "--preds_folder",
        type=str,
        default="True",
        help="예측 비디오 파일들이 있는 폴더 경로"
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=None,
        help="정답(타겟) 비디오 파일들이 있는 폴더 경로"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["fvd", "clip"],
        help="평가할 메트릭 리스트 (예: fvd clip)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="평가를 수행할 디바이스 (예: cpu 또는 cuda)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="각 비디오에서 로드할 프레임 수"
    )
    args = parser.parse_args()

    # # 폴더 존재 여부 확인
    # if not os.path.exists(args.preds_folder):
    #     print(f"예측 폴더 {args.preds_folder} 가 존재하지 않습니다.")
    #     exit(1)
    # if not os.path.exists(args.target_folder):
    #     print(f"타겟 폴더 {args.target_folder} 가 존재하지 않습니다.")
    #     exit(1)

    checkpoint_dir_list = ["20255", "40511", "60767", "81023", "101279"]

    for checkpoint in checkpoint_dir_list:
        args.metrics = "clip"
        args.device = "cuda:0"

        # args.preds_folder = f"/workspace/VideoCAM_MIIL/NEW_MMG_step_{checkpoint}_vggsound_sparse/video"        
        args.preds_folder = f"/workspace/VideoCAM_MIIL/MMG_NEW_NAIVE_DISTILL_step_{checkpoint}_vggsound_sparse/video"

        # 평가 실행
        fvd_value, clip_value = evaluate_video_metrics(
            args.preds_folder,
            args.target_folder,
            args.metrics,
            args.device,
            args.num_frames
        )

        # 결과 출력
        print("\n=== Video Metrics Evaluation Results ===")
        print(args.preds_folder)
        if "fvd" in args.metrics:
            print(f"FVD Score: {fvd_value}")
        if "clip" in args.metrics:
            print(f"CLIP Score: {clip_value}")

if __name__ == "__main__":
    main()
