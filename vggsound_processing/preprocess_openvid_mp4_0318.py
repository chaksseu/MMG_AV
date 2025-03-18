import os
import subprocess
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import tempfile

# ---------------------------
# 설정 변수
# ---------------------------
SOURCE_DIR = "/workspace/openvid_100_test" #'/workspace/openvid_1m_dataset/video'    # 원본 mp4 파일 폴더
DEST_DIR = '/workspace/data/preprocessed_openvid_videos_100_0318'  # 전처리된 결과 저장 폴더
MIN_DURATION = 3.6      # 최소 영상 길이(초)
MAX_DURATION = 30.0     # 최대 영상 길이(초)
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 320
OUTPUT_FPS = 12.5     # 최종 출력 FPS
NUM_WORKERS = 4

os.makedirs(DEST_DIR, exist_ok=True)

def get_video_duration(video_path):
    """ffprobe로 비디오 길이(초)를 반환 (실패시 None)"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None

def resize_and_center_crop_frame(frame, target_w, target_h):
    """
    원본 프레임의 비율을 유지하면서,
      1) target_w와 target_h를 모두 만족하도록 리사이즈
      2) 중앙을 기준으로 target_w x target_h로 크롭
    """
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2
    cropped_frame = resized_frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w]
    return cropped_frame

def process_video(source_path, dest_path):
    """
    1. 원본 비디오의 길이를 구한 후, MIN_DURATION 미만이면 스킵.
    2. MAX_DURATION을 초과하면 0 ~ MAX_DURATION 구간으로 트리밍.
    3. 트리밍된 영상 구간 전체(duration)에서, 시간 기반으로 균등 샘플링하여
       최종 프레임 수 = OUTPUT_FPS × duration 만큼의 프레임을 추출.
    4. 각 프레임은 리사이즈 + 센터 크롭 후 VideoWriter에 기록.
       → 이 방식은 원본의 시간 흐름에 맞추어 프레임을 추출하므로, 영상 속도가 변하지 않습니다.
    """
    if os.path.exists(dest_path):
        return True

    duration = get_video_duration(source_path)
    if duration is None or duration < MIN_DURATION:
        return False

    trim_duration = min(duration, MAX_DURATION)
    final_frame_count = int(round(OUTPUT_FPS * trim_duration))
    if final_frame_count < 1:
        return False

    # ffmpeg로 0 ~ trim_duration 구간만 추출 (스트림 복사)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        temp_path = tmp_file.name

    trim_cmd = [
        'ffmpeg', '-y',
        '-i', source_path,
        '-ss', '0',
        '-t', str(trim_duration),
        '-c', 'copy',
        temp_path
    ]
    try:
        subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    # OpenCV로 트리밍된 영상 읽기 및 시간 기반 프레임 샘플링
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dest_path, fourcc, OUTPUT_FPS, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # 전체 trim_duration 구간을 final_frame_count 프레임으로 균등 분할
    # 각 프레임의 타임스탬프 (초) 계산
    for i in range(final_frame_count):
        # final_frame_count가 1이면 0초, 아니라면 0~trim_duration 사이를 균등 분할
        if final_frame_count > 1:
            t_sec = i * (trim_duration / (final_frame_count - 1))
        else:
            t_sec = 0.0
        t_sec = min(t_sec, trim_duration)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        processed = resize_and_center_crop_frame(frame, RESIZE_WIDTH, RESIZE_HEIGHT)
        out.write(processed)

    cap.release()
    out.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return True

def process_task(args):
    src, dst = args
    return process_video(src, dst)

def main():
    video_files = [
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isfile(os.path.join(SOURCE_DIR, f)) and f.lower().endswith('.mp4')
    ]
    total_files = len(video_files)
    print(f"총 {total_files}개의 mp4 파일을 처리합니다.")

    tasks = [
        (os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, f))
        for f in video_files
    ]

    print(f"멀티프로세싱: {NUM_WORKERS}개 프로세스 사용")
    results = []
    with Pool(NUM_WORKERS) as p:
        for result in tqdm(p.imap_unordered(process_task, tasks), total=total_files):
            results.append(result)

    success_count = sum(results)
    print(f"처리 완료: {success_count}/{total_files}개 성공")

if __name__ == "__main__":
    main()
