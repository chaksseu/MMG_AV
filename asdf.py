# import os
# import re

# def rename_files_in_folder(folder):
#     # _proc_숫자_batch 부분과 확장자(mp4 또는 wav)를 찾는 정규표현식
#     pattern = re.compile(r'(.*)_proc_\d+_batch(\.mp4|\.wav)$', re.IGNORECASE)
    
#     for root, dirs, files in os.walk(folder):
#         for filename in files:
#             # 정규표현식과 매치되는지 확인
#             match = pattern.match(filename)
#             if match:
#                 # 그룹1: 기존이름, 그룹2: .확장자
#                 new_filename = match.group(1) + match.group(2)
#                 old_path = os.path.join(root, filename)
#                 new_path = os.path.join(root, new_filename)
#                 print(f"Renaming: {old_path} -> {new_path}")
#                 os.rename(old_path, new_path)

# # 처리할 폴더 경로 리스트 (필요에 따라 수정)
# folders = [
#     '/workspace/MMG_Inferencce_folder/0227_avsync_audio_teacher',
#     '/workspace/MMG_Inferencce_folder/0227_audio_teacher',
#     '/workspace/MMG_Inferencce_folder/0227_avsync_video_teacher',
#     '/workspace/MMG_Inferencce_folder/0227_video_teacher',
# ]

# for folder in folders:
#     rename_files_in_folder(folder)

import os
import re
from collections import defaultdict

# 정규표현식: {캡션}_batch_숫자.{확장자} (대소문자 구분 없이)
pattern = re.compile(r'^(.*?)_batch_\d+(\.mp4|\.wav)$', re.IGNORECASE)

# 캡션별 카운터: 각 캡션마다 0부터 시작해서 1씩 증가
counters = defaultdict(int)

# 검색할 최상위 디렉토리 경로 (사용자 환경에 맞게 변경)
base_dir = '/workspace/MMG_Inferencce_folder/0227_video_teacher'

# 최상위 디렉토리 및 모든 하위 디렉토리 탐색
for root, dirs, files in os.walk(base_dir):
    # 파일 순서를 일정하게 하기 위해 정렬 (선택 사항)
    files.sort()
    for filename in files:
        match = pattern.match(filename)
        if match:
            caption = match.group(1)  # 캡션 추출
            ext = match.group(2)      # 확장자 추출
            # 새 파일명: {캡션}_{카운터}.{확장자}
            new_filename = f"{caption}_{counters[caption]}{ext}"
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)
            print(f"Renaming: {old_path} -> {new_path}")  # 진행 상황 출력 (선택 사항)
            os.rename(old_path, new_path)
            counters[caption] += 1  # 해당 캡션의 카운터 증가
