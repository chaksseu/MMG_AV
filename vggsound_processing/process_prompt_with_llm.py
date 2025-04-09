import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import transformers
import torch
import pandas as pd
from tqdm import tqdm

# 모델 및 파이프라인 로드 (Meta-Llama-3.1-8B-Instruct)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
batch_size = 256
input_csv = "/home/work/kby_hgh/MMG_01/vggsound_processing/0401_video_llm_caption/videollama_split_3.csv"
output_csv = "/home/work/kby_hgh/MMG_01/vggsound_processing/0401_video_llm_caption/video_llm_splits/llm_videollama_split_3.csv"
prefix="A video of"
new_row_name="llm_video_caption"

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
                f"You are an assistant that rewrites prompts into a single English sentence starting with '{prefix} ~'. "
                "Your task is to preserve and accurately reflect **all factual information** from the original description."
                "Do **not** summarize, simplify, shorten, or omit any part of the content. "
                "Do **not** make up or infer details that are not explicitly stated in the original description. "
                f"Your output must be **factually faithful**, **grammatically correct**, and **structured as one complete sentence** that begins with '{prefix}'. "
                "Use commas and appropriate clauses to connect ideas clearly, even in long sentences."
            )
        },
        {
            "role": "user",
            "content": (
                f"Convert the following caption into a single sentence that begins with '{prefix}', while preserving **every detail exactly as described** and without adding any information not present in the original: {caption}"
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
df[new_row_name] = new_captions
df.to_csv(output_csv, index=False)
