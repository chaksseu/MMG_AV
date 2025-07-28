import os
import re
from collections import defaultdict

# 경로 설정
video_dir = "./video"
audio_dir = "./audio"

# 파일명에서 category 추출을 위한 정규식
pattern = re.compile(r"^(.*)_batch_\d+_proc_\d+_batch\.(mp4|wav)$")

# 파일 목록 정리 함수
def get_grouped_files(dir_path, ext):
    grouped = defaultdict(list)
    for f in os.listdir(dir_path):
        if not f.endswith(f".{ext}"):
            continue
        match = pattern.match(f)
        if match:
            category = match.group(1)
            grouped[category].append(f)
    # 각 그룹 내부 정렬
    for k in grouped:
        grouped[k].sort()
    return grouped

# video, audio 파일 그룹핑
video_groups = get_grouped_files(video_dir, "mp4")
audio_groups = get_grouped_files(audio_dir, "wav")

# 공통 카테고리만 처리
common_categories = set(video_groups.keys()) & set(audio_groups.keys())

for category in common_categories:
    videos = video_groups[category]
    audios = audio_groups[category]

    # 최소 개수 기준으로 자르기
    count = min(len(videos), len(audios))

    for i in range(count):
        new_name = f"{category}_{i}"
        
        old_video_path = os.path.join(video_dir, videos[i])
        new_video_path = os.path.join(video_dir, f"{new_name}.mp4")
        
        old_audio_path = os.path.join(audio_dir, audios[i])
        new_audio_path = os.path.join(audio_dir, f"{new_name}.wav")
        
        os.rename(old_video_path, new_video_path)
        os.rename(old_audio_path, new_audio_path)

        print(f"Renamed: {videos[i]} -> {new_name}.mp4")
        print(f"Renamed: {audios[i]} -> {new_name}.wav")
