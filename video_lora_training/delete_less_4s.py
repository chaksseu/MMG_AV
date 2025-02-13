import os
import subprocess
import concurrent.futures
from tqdm import tqdm

FOLDER_PATH = '/home/jupyter/preprocessed_WebVid_10M_train_videos_0130'  # 삭제할 파일들이 있는 폴더 경로
DURATION_THRESHOLD = 4.0  # 초 단위

def get_video_duration(video_path):
    """ffprobe를 사용하여 비디오 길이를 가져옴"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "format=duration", "-of", "csv=p=0", video_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(output) if output else 0
    except Exception:
        return 0  # 오류 발생 시 0초 처리 (즉, 삭제 대상)

def check_and_delete_video(video_path):
    """비디오 길이가 4초 이하인 경우 삭제하고, 삭제하면 1 반환, 아니면 0 반환"""
    duration = get_video_duration(video_path)
    if duration <= DURATION_THRESHOLD:
        try:
            os.remove(video_path)
            print(f"Deleted: {video_path} ({duration:.2f} sec)")
            return 1
        except Exception as e:
            print(f"Error deleting {video_path}: {e}")
    return 0

def process_videos(folder_path):
    """멀티스레딩을 사용하여 폴더 내의 mp4 파일을 병렬 처리하고, tqdm으로 진행 상황 표시"""
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp4")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # executor.map 을 tqdm으로 감싸 진행 상황을 시각화
        results = list(tqdm(executor.map(check_and_delete_video, video_files),
                            total=len(video_files),
                            desc="Processing videos"))
    
    # 삭제한 영상 개수 합산 후 출력
    deleted_count = sum(results)
    print(f"총 삭제된 영상 개수: {deleted_count}")

if __name__ == "__main__":
    process_videos(FOLDER_PATH)