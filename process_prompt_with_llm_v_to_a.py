import os
# os.environ['TRANSFORMERS_CACHE'] = '/workspace/transformers'
# os.environ['HF_HOME'] = '/workspace/huggingface_cache'

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import transformers
import torch
import pandas as pd
from tqdm import tqdm

# 모델 및 파이프라인 로드 (Meta-Llama-3.1-8B-Instruct)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
batch_size = 256
input_csv = "/home/work/kby_hgh/0411_processed_Openvid_train.csv" # 0411_processed_Openvid_train.csv
output_csv = "/home/work/kby_hgh/0506_processed_Openvid_train_with_audio_caption.csv"


llm = transformers.pipeline(
    task="text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    batch_size=batch_size,
)

# pad_token 설정 (문자열로!)
llm.tokenizer.pad_token = llm.tokenizer.eos_token

# CSV 파일 읽기
df = pd.read_csv(input_csv)
captions = df["new_caption"].tolist()

# 각 캡션에 대해 시스템과 유저 메시지 구성
messages = [
    [
        {
            "role": "system",
            "content": (
                "You are an assistant that converts detailed video scene descriptions into realistic audio captions. "
                "Your only job is to describe what a person would hear in the scene — not what they would see. "
                "Do not describe visual elements such as objects, actions, body language, clothing, colors, or lighting unless they make a sound. "
                "Only include details that correspond to actual audible events (e.g., footsteps, engine noise, speech, rustling leaves). "
                "Avoid inferring unhearable context or emotion — describe only what is explicitly audible. "
                "Your output must be a single, factually accurate, natural-sounding English sentence that fully captures the scene's auditory experience."
            )
        },
        {
            "role": "user",
            "content": (
                "Convert the following video description into a realistic audio caption. Only describe the sounds that would be heard in the scene.\n\n"
                f"Video description: {caption}"
            )
        }
    ]
    for caption in captions
]

# tqdm를 이용하여 배치 단위로 처리 (배치 사이즈 32)
outputs = []
for i in tqdm(range(0, len(messages), batch_size), desc="Processing batches"):
    batch_messages = messages[i:i+batch_size]
    batch_outputs = llm(batch_messages, max_new_tokens=256)
    outputs.extend(batch_outputs)

# 출력 형식에 따라 생성된 텍스트 추출 (출력 구조에 맞게 수정)
new_captions = []
for output in outputs:
    # 생성된 텍스트 추출 (strip()으로 앞뒤 공백 제거)
    new_caption = output[0]['generated_text'][-1]['content'].strip()
    new_captions.append(new_caption)

# 새 캡션을 DataFrame에 추가 후 CSV 저장
df["caption"] = new_captions
df.to_csv(output_csv, index=False)
