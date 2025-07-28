import os
# os.environ['TRANSFORMERS_CACHE'] = '/workspace/transformers'
# os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import transformers
import torch
import pandas as pd
from tqdm import tqdm

# 모델 및 파이프라인 로드 (Meta-Llama-3.1-8B-Instruct)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
batch_size = 1024
input_csv = "/home/work/kby_hgh/MMG_AC_test_dataset/0407_one_cap_AC_test.csv" # MMG_TA_dataset_filtered_0321.csv
output_csv = "/home/work/kby_hgh/MMG_AC_test_dataset/0407_one_cap_AC_test_with_video_caption.csv" # 0504_RC_MMG_TA_dataset_filtered_0321


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
captions = df["caption"].tolist()

# 각 캡션에 대해 시스템과 유저 메시지 구성
messages = [
    [
        {
            "role": "system",
            "content": (
                "You are an assistant that converts audio captions into visual video descriptions. "
                "Your job is to describe the most plausible visual scene that would naturally produce the given sound. "
                "You may reasonably infer visible subjects, objects, settings, and actions as long as they are clearly implied by the sound. "
                "Do not invent implausible or overly specific visual details, and avoid describing anything not reasonably connected to the audio. "
                "Describe the scene as one grammatically correct English sentence, focusing on what a person would likely see while hearing that sound."
            )
        },
        {
            "role": "user",
            "content": (
                "Now generate a visual video caption for the following sound description. Focus on what the viewer would likely see while hearing the sound.\n"
                f"Audio caption: {caption}"
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
df["new_caption"] = new_captions
df.to_csv(output_csv, index=False)
