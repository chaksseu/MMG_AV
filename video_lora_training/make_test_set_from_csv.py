import os
import pandas as pd
import shutil

# CSV 파일 경로 및 폴더 경로 설정
csv_path = "/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_test_with_audio_caption.csv"    # CSV 파일 경로
source_folder = "/home/work/kby_hgh/processed_OpenVid_1M_videos"  # 원본 파일이 있는 폴더 경로
test_folder = "/home/work/kby_hgh/MMG_01/video_lora_training/processed_OpenVid_1M_test_videos_42"  # Test 파일을 복사할 폴더 경로

# Test 폴더가 존재하지 않으면 생성
os.makedirs(test_folder, exist_ok=True)

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# split 열이 "test"인 행 필터링
test_files = df[df['split'] == 'test']['id'].tolist()

# 파일 복사
for file_name in test_files:
    source_file = os.path.join(source_folder, file_name)
    dest_file = os.path.join(test_folder, file_name)

    # 파일 존재 여부 확인 후 복사
    if os.path.exists(source_file):
        shutil.copy2(source_file, dest_file)
        print(f"Copied: {file_name} -> {test_folder}")
    else:
        print(f"File not found: {file_name}")
