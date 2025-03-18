import os
import subprocess
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tempfile

# 설정 변수
SOURCE_DIR = '/workspace/data/vggsound_train'  # 원본 mp4 파일이 있는 폴더 경로
DEST_DIR = '/workspace/data/preprocessed_VGGSound_train_videos_0313'  # 전처리된 파일을 저장할 폴더 경로
MIN_DURATION = 4             # 최소 길이 (초)
MAX_DURATION = 30            # 최대 길이 (초)
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 320
TARGET_FRAME_COUNT = 125     # 최종 출력 영상이 가지게 될 프레임 수
OUTPUT_FPS = 12.5            # VideoWriter를 위한 형식상 fps
num_workers = 32

# 결과 디렉토리가 없으면 생성
os.makedirs(DEST_DIR, exist_ok=True)

def get_video_duration(video_path):
    """비디오의 길이를 초 단위로 반환"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None

def resize_and_center_crop_frame(frame, target_w, target_h):
    """
    원본 프레임의 비율을 유지하면서,
    1) 가로 >= target_w, 세로 >= target_h 이상이 되도록 리사이즈
    2) 이후 (target_w, target_h) 크기로 센터 크롭
    """

    h, w = frame.shape[:2]

    # (1) 스케일 팩터 계산: 가로/세로가 각각 target_w, target_h 이상이 되도록
    scale = max(target_w / w, target_h / h)

    # 새로운 크기 계산
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # 리사이즈 (비율 유지)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # (2) 센터 크롭 (new_w x new_h) -> (target_w x target_h)
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2

    # 크롭 영역 계산 (중앙 기준)
    cropped_frame = resized_frame[y_offset:y_offset + target_h, x_offset:x_offset + target_w]

    return cropped_frame

def resize_video_fixed_frames(input_path, output_path, width, height, target_frame_count):
    """
    원본 영상을 읽고, 고정된 프레임 수(target_frame_count)만큼 균등 간격으로 샘플링하여
    종횡비 유지 -> 최소 (width, height) 이상으로 리사이즈 -> (width, height)로 센터 크롭 후 새 MP4로 저장.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False

    # fourcc, VideoWriter 준비
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (width, height))

    # 만약 원본 프레임 수가 목표 프레임 수보다 작다면:
    # - 모든 프레임을 사용하되, 부족하면 영상은 짧게 유지 (여기서는 단순히 프레임 수만큼만 저장)
    if total_frames <= target_frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # 비율 유지 + 센터크롭
            frame_processed = resize_and_center_crop_frame(frame, width, height)
            out.write(frame_processed)

    else:
        # 균등 간격으로 target_frame_count개의 프레임 인덱스 추출
        frame_indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            # 비율 유지 + 센터크롭
            frame_processed = resize_and_center_crop_frame(frame, width, height)
            out.write(frame_processed)

    cap.release()
    out.release()
    return True

def process_video(source_path, dest_path):
    """비디오 전처리 함수: 트림 후 리사이즈 (프레임 고정)"""
    
    # 이미 대상 파일이 존재하면 건너뛰기
    if os.path.exists(dest_path):
        return True

    duration = get_video_duration(source_path)
    if duration is None:
        return False
    if duration < MIN_DURATION:
        return False

    # 트림할 길이 결정
    trim_duration = min(duration, MAX_DURATION)

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        temp_path = tmp_file.name

    # ffmpeg 명령어 구성: 비디오 트림 (스트림 복사)
    trim_cmd = [
        'ffmpeg',
        '-y',
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

    # 리사이즈 + 프레임 고정
    success = resize_video_fixed_frames(
        temp_path,
        dest_path,
        width=RESIZE_WIDTH,
        height=RESIZE_HEIGHT,
        target_frame_count=TARGET_FRAME_COUNT
    )

    # 임시 파일 삭제
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return success

def process_task(task):
    """(source_path, dest_path) 튜플을 받아 process_video 실행"""
    source_path, dest_path = task
    return process_video(source_path, dest_path)

def main():
    # 모든 mp4 파일 목록 가져오기
    video_files = [
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isfile(os.path.join(SOURCE_DIR, f)) and f.lower().endswith('.mp4')
    ]

    total_files = len(video_files)
    print(f"총 {total_files}개의 mp4 파일을 처리합니다.")

    # 멀티프로세싱 설정
    pool_size = num_workers
    print(f"멀티프로세싱을 사용하여 {pool_size}개의 프로세스로 처리합니다.")

    # (소스경로, 대상경로) 튜플의 리스트 생성
    tasks = [
        (os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, f))
        for f in video_files
    ]

    results = []
    with Pool(pool_size) as pool:
        for result in tqdm(pool.imap_unordered(process_task, tasks), total=total_files):
            results.append(result)

    success_count = sum(results)
    print(f"처리 완료: {success_count}/{total_files} 파일 성공")

if __name__ == '__main__':
    main()
