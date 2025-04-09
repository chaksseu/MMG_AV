import os
import subprocess
import cv2
from multiprocessing import Pool
from tqdm import tqdm
import tempfile

# ---------------------------
# 설정 변수
# ---------------------------
SOURCE_DIR = '/home/work/kby_hgh/OpenVid_1M_download/video'  # 원본 mp4 파일 폴더
DEST_DIR = '/home/work/kby_hgh/processed_OpenVid_1M_videos'  # 전처리된 결과 저장 폴더
MIN_DURATION = 3.2      # 최소 영상 길이(초)
MAX_DURATION = 30.0     # 최대 영상 길이(초)
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 320
OUTPUT_FPS = 12.5     # 최종 출력 FPS
NUM_WORKERS = 48

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

def resize_preserving_aspect_ratio(frame, target_w, target_h):
    """
    원본 프레임의 비율을 유지하면서, 
    가로와 세로 중 최소한 하나가 target 크기 이상이 되도록 리사이즈.
    (즉, scale = max(target_w / 원본너비, target_h / 원본높이))
    """
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def process_video(source_path, dest_path):
    """
    1. 원본 비디오 길이를 확인하여 MIN_DURATION 미만이면 스킵.
    2. MAX_DURATION을 초과하면 0 ~ MAX_DURATION 구간으로 트리밍.
    3. 트리밍된 구간에서 시간 기반 균등 샘플링으로
       최종 프레임 수 = OUTPUT_FPS × (트리밍 길이) 만큼 프레임을 추출.
    4. 각 프레임에 대해 원본 비율 유지 리사이즈(최소 크기 이상) 후 VideoWriter에 기록.
       (VideoWriter는 첫 프레임의 리사이즈 결과 크기를 기준으로 초기화)
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

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    # 균등하게 샘플링한 타임스탬프에 따라 프레임 추출
    for i in range(final_frame_count):
        t_sec = i * (trim_duration / (final_frame_count - 1)) if final_frame_count > 1 else 0.0
        t_sec = min(t_sec, trim_duration)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        processed = resize_preserving_aspect_ratio(frame, RESIZE_WIDTH, RESIZE_HEIGHT)
        # 첫 프레임을 읽은 후, VideoWriter 초기화 (모든 프레임은 같은 크기로 기록)
        if out is None:
            height, width = processed.shape[:2]
            out = cv2.VideoWriter(dest_path, fourcc, OUTPUT_FPS, (width, height))
        out.write(processed)

    cap.release()
    if out is not None:
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
