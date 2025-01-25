import os
import subprocess
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# 설정 변수
SOURCE_DIR = 'WebVid_10M_train_videos'        # 원본 mp4 파일이 있는 폴더 경로
DEST_DIR = 'preprocessed_WebVid_10M_train_videos_0125'     # 전처리된 파일을 저장할 폴더 경로
MIN_DURATION = 4                           # 최소 길이 (초)
MAX_DURATION = 30                          # 최대 길이 (초)
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 320
TARGET_FPS = 12.5

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

def process_video(source_path, dest_path):
    """비디오 전처리 함수"""
    duration = get_video_duration(source_path)
    if duration is None:
        print(f"오류: 비디오 길이를 가져올 수 없습니다: {source_path}")
        return False
    if duration < MIN_DURATION:
        print(f"건너뜀 (길이 부족): {source_path} ({duration:.2f}초)")
        return False
    # 자를 길이 결정
    trim_duration = min(duration, MAX_DURATION)
    # ffmpeg 명령어 구성
    cmd = [
        'ffmpeg',
        '-y',  # 기존 파일 덮어쓰기
        '-i', source_path,
        '-ss', '0',  # 시작 시간
        '-t', str(trim_duration),  # 길이
        '-vf', f"scale={RESIZE_WIDTH}:{RESIZE_HEIGHT},fps={TARGET_FPS}",
        '-c:a', 'copy',  # 오디오 스트림 복사
        dest_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"오류: 처리 실패: {source_path}")
        return False

def main():
    # 모든 mp4 파일 목록 가져오기
    video_files = [
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isfile(os.path.join(SOURCE_DIR, f)) and f.lower().endswith('.mp4')
    ]

    total_files = len(video_files)
    print(f"총 {total_files}개의 mp4 파일을 처리합니다.")

    # 멀티프로세싱 설정
    pool_size = cpu_count() // 2
    print(f"멀티프로세싱을 사용하여 {pool_size}개의 프로세스로 처리합니다.")

    # 프로세스할 파일의 전체 경로와 대상 경로 준비
    tasks = [
        (os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, f))
        for f in video_files
    ]

    # 부분 함수 생성
    process_func = partial(process_video)

    # tqdm을 사용하여 진행 상황 표시
    with Pool(pool_size) as pool:
        results = list(tqdm(pool.starmap(process_video, tasks), total=total_files))

    success = sum(results)
    print(f"처리 완료: {success}/{total_files} 파일 성공")

if __name__ == '__main__':
    main()
