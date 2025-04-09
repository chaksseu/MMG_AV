import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 로그 경로 설정
log_dir = 'tensorboard/infer_no_lora_0402_MMG_1e-4_1219_1e-4_8gpu_abl_llm_captions_continue'

# 0404_MMG_1e-4_1e-4_8gpu_abl_combined_llm_caption
# 0404_MMG_1e-4_1e-4_8gpu_abl_videollama


# 이벤트 로딩
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# 저장 디렉토리 생성
output_dir = log_dir
os.makedirs(output_dir, exist_ok=True)

# 모든 스칼라 태그 가져오기
scalar_tags = event_acc.Tags()['scalars']

# 하나의 CSV 파일에 저장
csv_path = os.path.join(output_dir, "all_scalars.csv")
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["step", "tag", "value"])  # 헤더

    for tag in scalar_tags:
        scalars = event_acc.Scalars(tag)
        for s in scalars:
            writer.writerow([s.step, tag, s.value])

print(f"Saved all scalar data to CSV: {csv_path}")
