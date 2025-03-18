import os
import random
import shutil

# 원본 및 대상 폴더 경로
source_folder = "/workspace/openvid_1m_dataset/video"
dest_folder = "/workspace/openvid_100_test"

# 대상 폴더가 없으면 생성
os.makedirs(dest_folder, exist_ok=True)

# 비디오 파일 목록 가져오기
video_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# 파일이 100개 이상인지 확인
num_files_to_copy = min(100, len(video_files))
if num_files_to_copy == 0:
    print("No video files found in the source folder.")
else:
    # 랜덤하게 100개 선택
    selected_files = random.sample(video_files, num_files_to_copy)
    
    # 파일 복사
    for file_name in selected_files:
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy2(src_path, dest_path)
    
    print(f"Copied {num_files_to_copy} video files to {dest_folder}")