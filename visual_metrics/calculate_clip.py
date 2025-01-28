import numpy as np
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
from math import inf
import cv2
'''

def calculate_clip(videos1, caps, calculate_per_frame, calculate_final, clip_model, device):
    print("calculate_CLIP Score...")

    metric = CLIPScore(model_name_or_path=clip_model).to(device)
    videos1 = videos1.to(device)
    # videos [batch_size, timestamps, channel, h, w]
    clip_results = []
    idx = 0
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        text  = caps[video_num]
        video1 = videos1[video_num]

        #print(text)

        clip_results_of_a_video = []

        video1 = video1 * 255
        text = [text for i in range(len(video1))] 
        score = metric(video1, text)
        clip_results_of_a_video.append(score.item())


        clip_results.append(clip_results_of_a_video)
        

    clip_results = np.array(clip_results)

    clip = {}
    clip_std = {}

    if calculate_final:
        clip['final'] = np.mean(clip_results)
        clip_std['final'] = np.std(clip_results)

    result = {
        "clip": clip,
        "clip_std": clip_std,
        "clip_video_setting": video1.shape,
        "clip_video_setting_name": "time, channel, heigth, width",
    }

    return result
'''


import numpy as np
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm


def calculate_clip(
    videos1,
    caps,
    calculate_per_frame,
    calculate_final,
    clip_model,
    device,
    chunk_size=40,
):

    print("calculate_CLIP Score...")

    metric = CLIPScore(model_name_or_path=clip_model).to(device)

    clip_results = []

    for video_num in tqdm(range(videos1.shape[0])):
        text = caps[video_num]
        # [timestamps, channel, height, width]
        video_cpu = videos1[video_num]  
        video_cpu = video_cpu * 255

        scores_per_video = []

        for start_idx in range(0, video_cpu.shape[0], chunk_size):
            end_idx = start_idx + chunk_size
            frames_chunk = video_cpu[start_idx:end_idx]

            frames_chunk = frames_chunk.to(device)

            text_chunk = [text for _ in range(frames_chunk.shape[0])]

            with torch.no_grad():  
                score = metric(frames_chunk, text_chunk)

            scores_per_video.append(score.item())

            del frames_chunk
            torch.cuda.empty_cache()

        video_clip_score = np.mean(scores_per_video)
        clip_results.append(video_clip_score)

    clip_results = np.array(clip_results)

    clip_dict = {}
    clip_std_dict = {}

    if calculate_final:
        clip_dict['final'] = np.mean(clip_results)
        clip_std_dict['final'] = np.std(clip_results)

    result = {
        "clip": clip_dict,
        "clip_std": clip_std_dict,
        "clip_video_setting": videos1[0].shape,
        "clip_video_setting_name": "time, channel, height, width",
    }

    return result
