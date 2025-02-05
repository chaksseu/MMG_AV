import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random  # 랜덤 스타트 지점을 위한 import

# --- 설정 변수 ---
SOURCE_DIR = "/home/jupyter/preprocessed_WebVid_10M_gt_test_videos_500_0205"    # 원본 MP4가 있는 폴더
DEST_DIR   = "/home/jupyter/preprocessed_WebVid_10M_gt_test_videos_500_random_crop_0205_2"  # 최종 결과물이 저장될 폴더

TARGET_FPS = 12.5         # FPS를 12.5로 설정 (고정)
TRIM_DURATION = 3.2       # 잘라낼 길이(초) -> 3.2초
TOTAL_FRAMES = 40         # TRIM_DURATION * TARGET_FPS = 3.2 * 12.5 = 40 프레임
MINIMUM_LENGTH = 3.2      # 최소 길이 -> 3.2초보다 짧으면 스킵

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
    """
    1. 영상 길이를 확인한다.
    2. 길이가 3.2초 미만이라면 스킵한다.
    3. 0초 ~ (duration-3.2) 범위 내에서 랜덤한 start_time을 선택한다.
    4. ffmpeg로 3.2초 구간만 자르고, FPS=12.5로 인코딩하여 저장한다.
    """
    # 이미 대상 파일이 존재하면 건너뛰기
    if os.path.exists(dest_path):
        return True

    # 비디오 길이 확인
    duration = get_video_duration(source_path)
    if duration is None:
        return False
    if duration < MINIMUM_LENGTH:
        return False

    # 랜덤 시작점 (duration - 3.2) 내에서 무작위 추출
    start_time = random.uniform(0, duration - TRIM_DURATION)

    # ffmpeg 명령어 구성
    trim_cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-i', source_path,
        "-frames:v", str(TOTAL_FRAMES),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        dest_path
    ]
    try:
        subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        return False

    return True

# 전역 범위에서 사용할 wrapper 함수 정의
def process_video_wrapper(args):
    """Multiprocessing 사용 시 인자를 unpacking하여 process_video를 호출"""
    return process_video(*args)

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
        # lambda 대신 전역의 process_video_wrapper 함수를 사용
        for result in tqdm(pool.imap_unordered(process_video_wrapper, tasks), total=total_files):
            results.append(result)

    success_count = sum(results)
    print(f"처리 완료: {success_count}/{total_files} 파일 성공")

if __name__ == '__main__':
    main()
