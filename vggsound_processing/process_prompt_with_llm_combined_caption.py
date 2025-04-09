import os

split_num = 7

os.environ["CUDA_VISIBLE_DEVICES"] = f"{split_num}"

import transformers
import torch
import pandas as pd
from tqdm import tqdm

# 모델 및 파이프라인 로드 (예시: Meta-Llama-3.1-8B-Instruct)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
batch_size = 32  # 필요에 따라 조정
input_csv = f"/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs/videollama_split_{split_num}.csv"   # "audio_caption", "video_caption" 열이 존재한다고 가정
output_csv = f"/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs/generated_csvs/combined_split_{split_num}.csv" # 결과를 저장할 csv
new_row_name = "combined_caption"

# 모델 로드 (transformers Pipeline 예시)
llm = transformers.pipeline(
    task="text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    batch_size=batch_size,
)

# PAD 토큰 설정
llm.tokenizer.pad_token = llm.tokenizer.eos_token

# CSV 파일 읽기
df = pd.read_csv(input_csv)

# 오디오/비디오 캡션 리스트로 가져오기
audio_captions = df["caption"].tolist()
video_captions = df["new_caption"].tolist()

# 각 행의 (audio_caption, video_caption) 쌍을 합쳐 메시지 구성
messages = []
for audio_cap, video_cap in zip(audio_captions, video_captions):
    system_message = {
        "role": "system",
        "content": (
            "You are an assistant that combines an audio caption and a video caption into a single English description. "
            "Your task is to precisely merge all factual information from both the audio and video descriptions. "
            "Do not omit any details and do not add or infer anything that is not explicitly stated. "
            "Your final answer must be grammatically correct and faithful to the original content. "
            "Use commas and appropriate connectors to naturally integrate both captions."
        )
    }
    user_message = {
        "role": "user",
        "content": (
            "Combine the following audio caption and video caption into a single coherent description, "
            "without losing or altering any information:\n\n"
            f"Audio Caption: {audio_cap}\n"
            f"Video Caption: {video_cap}"
        )
    }
    messages.append([system_message, user_message])

# 배치 처리를 위한 루프
outputs = []
for i in tqdm(range(0, len(messages), batch_size), desc="Processing batches"):
    batch_messages = messages[i:i + batch_size]
    batch_outputs = llm(batch_messages, max_new_tokens=256)
    outputs.extend(batch_outputs)

# 결과에서 실제 생성 텍스트를 추출
new_captions = []
for output in outputs:
    # 사용 중인 파이프라인 출력 형식에 맞춰 텍스트 추출
    # (아래는 예시로 'generated_text' 키를 사용)
    new_caption = output[0]['generated_text'][-1]['content'].strip()
    new_captions.append(new_caption)
    # # generated_text = output[0]["generated_text"]
    # # combined_text = generated_text.strip()
    # new_captions.append(combined_text)

# DataFrame에 새로운 열로 추가
df[new_row_name] = new_captions

# 결과 CSV 저장
df.to_csv(output_csv, index=False)
