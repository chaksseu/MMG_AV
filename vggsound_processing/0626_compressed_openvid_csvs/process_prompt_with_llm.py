split_num = 7


import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{split_num}"

import transformers
import torch
import pandas as pd
from tqdm import tqdm

# 모델 및 파이프라인 로드 (Meta-Llama-3.1-8B-Instruct)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
batch_size = 128
input_csv = f"/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/original_{split_num}.csv"
output_csv = f"/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_{split_num}.csv"
# input_csv = "/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_test_with_audio_caption_filtered.csv"
# output_csv = "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/0626_processed_Openvid_test_with_audio_caption_filtered_summerize.csv"
new_row_name="compressed_video_caption"

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
                "You are an assistant that compresses video descriptions into concise versions around 30% of the original length. "
                "Keep only the most essential information — such as key subjects, actions, and settings. "
                "Remove redundant adjectives, stylized language, and non-essential details. "
                "Do not add or infer information that is not explicitly mentioned. "
                "Your output must be factually accurate, grammatically correct, and concise while preserving the main ideas."
            )
        },
        {
            "role": "user",
            "content": (
                "Compress the following video description to approximately 30% of its original length. "
                "Preserve key facts and core content, and eliminate verbose or decorative language.\n\n"
                f"Text:\n\"{caption}\""
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
