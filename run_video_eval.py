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

def load_videos_with_caps(folder_path, num_frames):
    videos_tensor_list_orig = []
    caps = []
    # open and read the file
    vid_fnames = sorted(os.listdir(folder_path))
    for video_name in vid_fnames:
        fname = video_name[:-4]
        caps.append(clean_sentence(fname))
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

    calculate_final = True
    calculate_per_frame = 8

    if 'fvd' in metrics:
        orig_videos = load_videos_from_folder(target_folder, num_frames)
        new_videos = load_videos_from_folder(preds_folder, num_frames)
        fvd_score = calculate_fvd(orig_videos, new_videos, calculate_per_frame, calculate_final, device)

    if 'clip' in metrics:
        clip_videos, caps = load_videos_with_caps(preds_folder, num_frames)
        clip_score = calculate_clip(clip_videos, caps, calculate_per_frame, calculate_final, args.clip_model, device)

    return fvd_score["fvd"], clip_score["clip"]['final']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate video on visual metrics')

    # Paths
    parser.add_argument('--target_folder', required=True, type=str,
                        help='Path to original videos.')
    parser.add_argument('--preds_folder', required=True, type=str,
                        help='Path to new videos.')

    # Metrics related arguments
    parser.add_argument('--metrics', nargs='+', type=str,
                        default=['fvd', 'ssim', 'psnr', 'psnrb', 'lpips', 'ms_ssim', 'clip', 'fid', 'kid',
                                 'mifid', 'vmaf', 'vif'],
                        help='Metrics to compute. Possible values: fvd, ssim, psnr, psnrb, lpips, ms_ssim, clip, fid, kid, mifid, vmaf, vif')
    #parser.add_argument('--net', default='alex', type=str,
    #                    help="Backbone network type for lpips. Choose between 'alex', 'vgg', or 'squeeze'")
    parser.add_argument('--clip_model', default='openai/clip-vit-base-patch16', type=str,
                        help='Version of the CLIP model to use. Available models: "openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14"')
    parser.add_argument('--feat_layer', default=64, type=int,
                        help='Inceptionv3 feature layer for FID, KID, IS, MIFID. Options: 64, 192, 768, 2048')

    # Frame related arguments
    parser.add_argument('--num_frames', default=40, type=int, help='Number of frames.')
    #parser.add_argument('--subset_size', default=16, type=int,
    #                    help='Frame samples in each video for KID computation')
    parser.add_argument('--calculate_per_frame', default=8, type=int,
                        help='Calculation per frame.')

    # Miscellaneous
    parser.add_argument('--results_file', default='results.txt', type=str,
                        help='File to save the results.')
    parser.add_argument('--calculate_final', default=True, action='store_true',
                        help='Calculate final metrics.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Device for computations. Default is cuda.')

    args = parser.parse_args()
    main(args)