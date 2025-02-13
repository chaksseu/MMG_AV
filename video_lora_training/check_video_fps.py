import os
import logging
from moviepy.editor import VideoFileClip

# 로그 설정
logging.basicConfig(
    filename="fps_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_video_fps(directory):
    """
    특정 폴더 아래의 모든 MP4 파일들의 FPS를 로깅하는 함수
    """
    if not os.path.isdir(directory):
        logging.error(f"디렉토리를 찾을 수 없습니다: {directory}")
        return
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp4"):
                file_path = os.path.join(root, file)
                try:
                    with VideoFileClip(file_path) as clip:
                        fps = clip.fps
                        logging.info(f"파일: {file_path}, FPS: {fps}")
                        print(f"파일: {file_path}, FPS: {fps}")
                except Exception as e:
                    logging.error(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")

# 실행 예제
target_folder = "/home/jupyter/preprocessed_WebVid_10M_gt_test_videos_1k_random_crop_0210"  # 확인할 폴더 경로 지정
log_video_fps(target_folder)
