import os
import cv2

def count_frames(video_path):
    """동영상 파일의 프레임 개수를 반환"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None  # 파일 열기 실패 시 None 반환
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def log_frame_counts(folder_path, log_file="0305_frame_log.txt", low_frame_log="low_frame_videos.txt"):
    """폴더 내 MP4 파일의 프레임 개수를 측정하여 로깅, 40보다 작은 파일을 별도 저장"""
    with open(log_file, "w") as log, open(low_frame_log, "w") as low_log:
        log.write("파일명, 프레임 개수\n")
        low_log.write("프레임 개수 40 미만 파일 목록\n")
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    frame_count = count_frames(video_path)
                    
                    if frame_count is not None:
                        log.write(f"{file}, {frame_count}\n")
                        
                        if frame_count < 125:
                            low_log.write(f"{file}, {frame_count}\n")
                            print(f"[경고] {file}: 프레임 개수 {frame_count} (40 미만)")
                        # elif frame_count != 40:
                        #     print(f"[알림] {file}: 프레임 개수 {frame_count} (40과 다름)")
                    else:
                        print(f"[오류] {file}: 프레임 개수 확인 실패")

    print(f"프레임 개수 로깅 완료: {log_file}")
    print(f"프레임 개수 40 미만 파일 로깅 완료: {low_frame_log}")

# 사용 예시
folder_path = "/workspace/data/preprocessed_VGGSound_train_videos_0310"  # 여기에 폴더 경로 입력
log_frame_counts(folder_path)
