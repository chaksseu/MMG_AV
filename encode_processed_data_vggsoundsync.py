import torch
from diffusers import PNDMScheduler, UNet2DConditionModel
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer, CrossModalTransformer,
    Downsample, Upsample, TimestepBlock
)
from lvdm.common import checkpoint
from einops import rearrange
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

import torch.nn as nn



import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from auffusion_pipe_functions import (
    _execution_device, _encode_prompt, prepare_extra_step_kwargs, 
    prepare_latents, ConditionAdapter, import_model_class_from_model_name_or_path
)


from processed_data_dataloader import VideoAudioTextDataset

import os
import torch
import csv
from pathlib import Path

# 데이터를 저장할 기본 폴더 설정
OUTPUT_DIR = "latents_data_32s_40frames_vggsoundsync_new_normalization"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video_latents")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_latents")
VIDEO_TEXT_DIR = os.path.join(OUTPUT_DIR, "video_text_embeds")
AUDIO_TEXT_DIR = os.path.join(OUTPUT_DIR, "audio_text_embeds")

CSV_FILE = os.path.join(OUTPUT_DIR, "dataset_info.csv")


# 폴더 생성 함수
def create_output_directory():
    """
    출력 폴더를 생성합니다.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(VIDEO_DIR).mkdir(parents=True, exist_ok=True)
    Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    Path(VIDEO_TEXT_DIR).mkdir(parents=True, exist_ok=True)
    Path(AUDIO_TEXT_DIR).mkdir(parents=True, exist_ok=True)


# CSV 파일 생성 함수
def create_csv_file():
    """
    CSV 파일을 생성합니다.
    """
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Video", "Audio", "Video_text_embed", "Audio_text_embed"])
        print(f"CSV 파일 생성: {CSV_FILE}")

def save_to_csv(Video, Audio, video_text, audio_text):
    """
    파일 이름과 텍스트 정보를 CSV 파일에 저장합니다.
    """
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([Video, Audio, video_text, audio_text])
    print(f"CSV에 데이터 저장 - Video: {Video}, Audio: {Audio}, Video_Text: {video_text}, Audio_Text: {audio_text}")


def save_item(video: torch.Tensor, audio: torch.Tensor, video_text_embed: torch.Tensor, audio_text_embed: torch.Tensor, index: int):
    """
    전처리된 데이터를 디스크에 저장하고 CSV 파일에 파일 이름과 텍스트 정보를 기록합니다.
    """

    v = f"video_{index}.pt"
    a = f"audio_{index}.pt"
    vt = f"video_text_embed_{index}.pt"
    at = f"audio_text_embed_{index}.pt"
    video_save_path = None
    audio_save_path = None
    video_text_save_path = None
    audio_text_save_path = None

    # save video
    if not video == None:
        video_save_path = os.path.join(VIDEO_DIR, v)
        torch.save(video, video_save_path)
    # save audio
    if not audio == None:
        audio_save_path = os.path.join(AUDIO_DIR, a)
        torch.save(audio, audio_save_path)
    # save video text
    if not video == None:
        video_text_save_path = os.path.join(VIDEO_TEXT_DIR, vt)
        torch.save(video_text_embed, video_text_save_path)
    # save audio text
    if not audio == None:
        audio_text_save_path = os.path.join(AUDIO_TEXT_DIR, at)
        torch.save(audio_text_embed, audio_text_save_path)

    save_to_csv(v, a, vt, at)

    print(f"Complete saving data - video: {video_save_path}, audio: {audio_save_path}, video_text: {video_text_save_path}, audio_text: {audio_text_save_path}")



def main():
    args = parse_args()
    device = "cuda:1"
    dtype = torch.float32
    create_output_directory()
    create_csv_file()

    csv_file = "../_processed_data_32s_40frame_vggsoundsync_new_normalization/dataset_info.csv"
    video_folder = "../_processed_data_32s_40frame_vggsoundsync_new_normalization/video"
    audio_folder = "../_processed_data_32s_40frame_vggsoundsync_new_normalization/audio"



    dataset = VideoAudioTextDataset(csv_path=csv_file, 
                                    video_dir=video_folder,  # Videos shape: torch.Size([B, 64, 3, 256, 256])
                                    audio_dir=audio_folder,  # Audios shape: torch.Size([B, 3, 256, 400])
                                    transform=None)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)



    ########################################## Prepare audio encoding ##########################################

    generator = torch.Generator(device=device).manual_seed(args.seed)
    pretrained_model_name_or_path = (
        snapshot_download(args.pretrained_model_name_or_path)
        if not os.path.isdir(args.pretrained_model_name_or_path) 
        else args.pretrained_model_name_or_path
    )
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device, dtype).eval()        
    vae.requires_grad_(False)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    condition_json_path = os.path.join(pretrained_model_name_or_path, "condition_config.json")
    condition_json_list = json.loads(open(condition_json_path).read())
    text_encoder_list, tokenizer_list, adapter_list = [], [], []

    with torch.no_grad():
        for i, condition_item in enumerate(condition_json_list):
            text_encoder_path = os.path.join(pretrained_model_name_or_path, condition_item["text_encoder_name"])
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            tokenizer_list.append(tokenizer)
    
            text_encoder_cls = import_model_class_from_model_name_or_path(text_encoder_path)
            text_encoder = text_encoder_cls.from_pretrained(text_encoder_path).to(device, dtype)
            text_encoder.requires_grad_(False)
            text_encoder_list.append(text_encoder)
    
            adapter_path = os.path.join(pretrained_model_name_or_path, condition_item["condition_adapter_name"])
            adapter = ConditionAdapter.from_pretrained(adapter_path).to(device, dtype)
            adapter.requires_grad_(False)
            adapter_list.append(adapter)

    ########################################## Prepare video encoding ##########################################


    video_config = OmegaConf.load('configs/inference_t2v_512_v2.0.yaml')
    video_model = instantiate_from_config(video_config.model)
    video_model.load_state_dict(torch.load('scripts/evaluation/model.ckpt')['state_dict'], strict=True)
    video_model.eval().to(device)
    

    for batch_idx, (videos, audios, texts) in enumerate(tqdm(dataloader, desc="Encoding batches")):
        batch_video = videos.to(device)
        batch_video = batch_video.permute(0, 2, 1, 3, 4).to(device)  # torch.Size([B, c=3, t=64, 256, 256])
        batch_audio = audios.to(device)
        batch_text = list(texts)

        print("batch_video", batch_video.shape)
        print("batch_audio", batch_audio.shape)
        print("batch_text", batch_text)

        ########################################## Encoding audio ##########################################
        spectrograms = [(spectrogram + 1) / 2 for spectrogram in batch_audio]
        image = image_processor.preprocess(spectrograms)
            
        with torch.no_grad():
            audio_text_embed = _encode_prompt(
                text_encoder_list=text_encoder_list,
                tokenizer_list=tokenizer_list,
                adapter_list=adapter_list,
                tokenizer_model_max_length=77,
                dtype=dtype,
                prompt=batch_text,
                device=device
            )
            audio_latent = prepare_latents(vae, image, args.batch_size, 1, audio_text_embed.dtype, device, generator)

        audio_latent = audio_latent.squeeze(dim=0)
        audio_text_embed = audio_text_embed.squeeze(dim=0)

        print("audio_latent", audio_latent.shape)
        print("audio_text_embed", audio_text_embed.shape)        


        ########################################## Encoding video ##########################################
        with torch.no_grad():
            video_latent = video_model.encode_first_stage(batch_video)
            video_text_embed = video_model.get_learned_conditioning(batch_text)

        video_latent = video_latent.squeeze(dim=0)
        video_text_embed = video_text_embed.squeeze(dim=0)
        print("video_latent", video_latent.shape)
        print("video_text_embed", video_text_embed.shape)        

        save_item(video=video_latent, audio=audio_latent, video_text_embed=video_text_embed, audio_text_embed=audio_text_embed, index=batch_idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Encode video & audio data")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="auffusion/auffusion-full", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inference.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    return parser.parse_args()



if __name__ == "__main__":
    main()
