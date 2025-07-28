import os
from collections import defaultdict

# 폴더 경로 설정
folder_path = "/home/work/kby_hgh/MMG_01/video_lora_training/processed_OpenVid_2000_test_videos_42"
# ID별 파일 목록 수집
id_to_files = defaultdict(list)

for filename in os.listdir(folder_path):
    if "_batch_" in filename:
        base_id = filename.split("_batch_")[0] + ".mp4"
        id_to_files[base_id].append(filename)

# 중복된 ID 출력
print("⚠️ 동일 ID를 갖는 파일이 2개 이상 있는 경우:")
for base_id, files in id_to_files.items():
    if len(files) > 1:
        print(f"{base_id}: {len(files)}개 파일 → {files}")