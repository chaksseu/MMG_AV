import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

# --- 설정 변수 ---
SOURCE_DIR = "/home/work/kby_hgh/processed_OpenVid_1M_videos"
DEST_DIR = "/home/work/kby_hgh/MMG_01/video_lora_training/processed_OpenVid_1920_test_full_videos_0627"
TARGET_FPS = 12.5
CSV_PATH = "/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_test_with_audio_caption_filtered.csv"

os.makedirs(DEST_DIR, exist_ok=True)

# --- CSV ID 목록 불러오기 ---
df = pd.read_csv(CSV_PATH)
id_set = set(df['id'].tolist())  # 예: ['abc123.mp4', 'xyz456.mp4']

def get_video_duration(video_path):
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

def process_video(source_path, dest_path, filename):
    if filename not in id_set:
        return False  # CSV에 없는 파일은 패스

    if os.path.exists(dest_path):
        return True

    duration = get_video_duration(source_path)
    if duration is None:
        return False

    cmd = [
        'ffmpeg',
        '-y',
        '-i', source_path,
        '-vf', f'fps={TARGET_FPS}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        dest_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def process_video_wrapper(args):
    return process_video(*args)

def main():
    video_files = [
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isfile(os.path.join(SOURCE_DIR, f)) and f.lower().endswith('.mp4')
    ]

    print(f"총 {len(video_files)}개의 mp4 파일 중, CSV 지정된 ID만 처리합니다.")
    tasks = [
        (os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, f), f)
        for f in video_files
    ]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_video_wrapper, tasks), total=len(tasks)))

    success_count = sum(results)
    print(f"처리 완료: {success_count}/{len(id_set)} (CSV 기준)")

if __name__ == '__main__':
    main()
