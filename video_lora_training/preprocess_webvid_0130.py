import os
import subprocess
import cv2
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tempfile

# 설정 변수
SOURCE_DIR = '/home/jupyter/WebVid_10M_train_videos'        # 원본 mp4 파일이 있는 폴더 경로
DEST_DIR = '/home/jupyter/preprocessed_WebVid_10M_train_videos_0130'     # 전처리된 파일을 저장할 폴더 경로
MIN_DURATION = 4                           # 최소 길이 (초)
MAX_DURATION = 30                          # 최대 길이 (초)
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 320
TARGET_FPS = 12.5  # FPS를 12.5로 설정

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

def resize_video(input_path, output_path, width=256, height=256, target_fps=30):
    """
    OpenCV(cv2)를 사용하여 MP4를 읽고 width x height 로 리사이즈 후 새 MP4로 저장.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = target_fps  # FPS가 0으로 읽히는 경우 임시 대처

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))
        out.write(frame_resized)

    cap.release()
    out.release()
    return True

def process_video(source_path, dest_path):
    """비디오 전처리 함수: 트림 후 리사이즈"""
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

    # ffmpeg 명령어 구성: 비디오 트림
    trim_cmd = [
        'ffmpeg',
        '-y',  # 기존 파일 덮어쓰기
        '-i', source_path,
        '-ss', '0',              # 시작 시간
        '-t', str(trim_duration), # 잘라낼 길이
        '-c', 'copy',            # 스트림 복사 (인코딩 X)
        temp_path
    ]
    try:
        subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    # 리사이즈
    success = resize_video(temp_path, dest_path, width=RESIZE_WIDTH, height=RESIZE_HEIGHT, target_fps=TARGET_FPS)

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
    pool_size = cpu_count()
    print(f"멀티프로세싱을 사용하여 {pool_size}개의 프로세스로 처리합니다.")

    # (소스경로, 대상경로) 튜플의 리스트 생성
    tasks = [
        (os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, f))
        for f in video_files
    ]

    results = []
    with Pool(pool_size) as pool:
        # imap_unordered: 각 작업이 완료될 때마다 바로 결과 반환
        for result in tqdm(pool.imap_unordered(process_task, tasks), total=total_files):
            results.append(result)

    success_count = sum(results)
    print(f"처리 완료: {success_count}/{total_files} 파일 성공")

if __name__ == '__main__':
        main()
