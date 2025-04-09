import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 예: 0번과 1번 GPU만 사용

# CSV 파일과 동영상 파일 경로 설정
csv_file = "/home/work/kby_hgh/MMG_01/vggsound_processing/0330_vgg_split_csvs/split_7_1.csv"         # 원본 CSV 파일 경로 (예: "data.csv")
video_folder = "/home/work/kby_hgh/workspace/data/preprocessed_VGGSound_train_no_crop_videos_0329"  # 동영상 파일들이 있는 폴더 경로
new_csv_file = "/home/work/kby_hgh/MMG_01/vggsound_processing/0330_vgg_split_csvs/video_llm_split_csvs/videollama3_new_captions_7_1.csv"


# 모델 및 프로세서 초기화
model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    # device=1,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

def caption_video(video_path):
    """
    주어진 video_path에 대해 캡셔닝을 수행하는 함수.
    """
    question = "Describe this video in detail."
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 12, "max_frames": 128}},
                {"type": "text", "text": question},
            ]
        },
    ]
    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 각 행에 대해 캡셔닝 수행
from tqdm import tqdm

new_captions = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Captioning Videos"):
    video_file = os.path.join(video_folder, f"{row['id']}.mp4")
    print(f"Processing video: {video_file}")
    caption = caption_video(video_file)
    new_captions.append(caption)
    print(f"Caption: {caption}")


# 새로운 캡셔닝 열 추가
df["new_caption"] = new_captions

# 새로운 CSV 파일로 저장
df.to_csv(new_csv_file, index=False)
print(f"New CSV saved to {new_csv_file}")
