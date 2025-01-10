import os
import cv2
import subprocess
import random
from pydub import AudioSegment

###############################################################################
# 0. 사용자 지정 경로 & 파라미터
###############################################################################
ORIGINAL_FOLDER = "vggsound_sparse_train"  # 원본 MP4가 있는 폴더
OUTPUT_FOLDER = "0110_vggsound_sparse_train_random_final"  # 최종 결과물이 저장될 폴더

TRIM_LENGTH = 3.2           # 잘라낼 길이(초)
TARGET_FRAME_COUNT = 40     # 최종 영상의 총 프레임 수 (40장)
RESIZE_WIDTH = 520
RESIZE_HEIGHT = 320

# 오디오 / 영상 / 합쳐진 영상 저장 폴더 (최종 결과)
AUDIO_FOLDER = os.path.join(OUTPUT_FOLDER, "audio")
VIDEO_FOLDER = os.path.join(OUTPUT_FOLDER, "video")
COMBINED_FOLDER = os.path.join(OUTPUT_FOLDER, "combined_video")

###############################################################################
# 1. 헬퍼 함수들
###############################################################################

def delete_files_in_folder(folder_path):
    """
    최상위 폴더의 파일만 삭제하고 하위 폴더 및 하위 폴더의 파일은 유지하는 함수.

    :param folder_path: 파일을 삭제할 최상위 폴더 경로
    """
    # 최상위 폴더의 파일 목록 확인
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # 파일인 경우 삭제 (폴더는 건너뜀)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                print(f"파일 삭제: {item_path}")
            except Exception as e:
                print(f"파일 삭제 실패: {item_path} - {e}")
        else:
            print(f"폴더 유지: {item_path}")

def get_video_duration(video_path):
    """
    ffprobe를 사용하여 동영상 길이를(초 단위 float) 반환합니다.
    """
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration_str = result.stdout.strip()
    try:
        return float(duration_str)
    except ValueError:
        return None

def trim_video(input_path, output_path, start_second, duration):
    """
    ffmpeg로 특정 구간(start_second ~ start_second+duration)을 잘라냄.
    """
    trim_cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_second}",
        "-i", input_path,
        "-t", f"{duration:.3f}",
        "-c:v", "copy",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def set_frame_count_to_target(input_path, output_path, target_frame_count):
    """
    동영상을 target_frame_count (예: 40프레임)가 되도록 fps를 재설정 후 저장합니다.
    새 fps = target_frame_count / 영상길이
    """
    duration = get_video_duration(input_path)
    if not duration or duration <= 0:
        print(f"[경고] 영상 길이를 구할 수 없어서 FPS 조정 불가: {input_path}")
        return False

    new_fps = target_frame_count / duration
    convert_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", f"fps={new_fps:.6f}",
        "-frames:v", str(target_frame_count),  # 정확히 40프레임으로 제한
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 프레임 수 검증
    actual_frames = count_frames(output_path)
    if actual_frames != target_frame_count:
        print(f"[경고] {output_path}의 프레임 수가 예상과 다릅니다: {actual_frames}프레임")
        return False
    return True

def count_frames(video_path):
    """
    OpenCV를 사용하여 동영상의 총 프레임 수를 반환합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[에러] 동영상을 열 수 없습니다: {video_path}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def extract_audio(input_path, output_path):
    """
    pydub 라이브러리를 사용하여 MP4 파일에서 오디오를 WAV로 추출.
    """
    try:
        audio = AudioSegment.from_file(input_path, format="mp4")
        audio.export(output_path, format="wav")
        print(f"[오디오 추출 완료] {output_path}")
    except Exception as e:
        print(f"[에러] 오디오 추출 실패: {input_path}, 오류: {e}")

def resize_video(input_path, output_path, width=256, height=256):
    """
    OpenCV(cv2)를 사용하여 MP4를 읽고 width x height 로 리사이즈 후 새 MP4로 저장.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fps가 0으로 읽히는 경우 임시 대처

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))
        out.write(frame_resized)
        frame_count += 1

    cap.release()
    out.release()
    print(f"[영상 리사이즈 완료] {output_path} ({frame_count}프레임)")

def combine_video_audio(video_path, audio_path, output_path):
    """
    ffmpeg로 리사이즈된 mp4와 wav 오디오를 합쳐 최종 mp4를 생성.
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # 영상은 재인코딩 없이 복사
        '-c:a', 'aac',   # 오디오는 aac로 인코딩
        output_path
    ]
    subprocess.run(command, check=True)
    print(f"[영상+오디오 합치기 완료] {output_path}")

###############################################################################
# 2. 메인 처리 함수
###############################################################################
def main():
    # 결과 저장 폴더 생성
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    os.makedirs(COMBINED_FOLDER, exist_ok=True)

    # ORIGINAL_FOLDER 안의 모든 MP4 파일을 가져옴
    mp4_files = [
        f for f in os.listdir(ORIGINAL_FOLDER)
        if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(ORIGINAL_FOLDER, f))
    ]

    for video_name in mp4_files:
        input_path = os.path.join(ORIGINAL_FOLDER, video_name)
        if not os.path.isfile(input_path):
            print(f"[경고] 파일이 존재하지 않음: {input_path}")
            continue

        # 전체 길이 확인
        total_duration = get_video_duration(input_path)
        if not total_duration or total_duration <= 0:
            print(f"[경고] 영상 길이를 구할 수 없음, 스킵: {video_name}")
            continue

        # 랜덤 시작 시간 설정 (영상이 TRIM_LENGTH보다 길면 그 안에서 랜덤으로 잘라냄)
        if total_duration > TRIM_LENGTH:
            trim_start = random.uniform(0, total_duration - TRIM_LENGTH)
            trim_end = trim_start + TRIM_LENGTH
        else:
            trim_start = 0
            trim_end = total_duration

        actual_length = trim_end - trim_start

        # 저장할 파일 이름(확장자 제외)
        base_name, _ = os.path.splitext(video_name)

        # 중간·최종 결과 파일 경로
        trimmed_path  = os.path.join(OUTPUT_FOLDER, f"{base_name}_trimmed.mp4")
        frame40_path  = os.path.join(OUTPUT_FOLDER, f"{base_name}_40frames.mp4")
        audio_path    = os.path.join(AUDIO_FOLDER,  f"{base_name}.wav")
        resized_path  = os.path.join(VIDEO_FOLDER,   f"{base_name}.mp4")  
        combined_path = os.path.join(COMBINED_FOLDER, f"{base_name}_combined.mp4")

        ###################################################################
        # (1) 3.2초 구간 잘라내기
        ###################################################################
        print(f"\n[1] Trimming {video_name}: {trim_start:.2f} ~ {trim_end:.2f}")
        trim_video(input_path, trimmed_path, trim_start, actual_length)

        ###################################################################
        # (2) 40프레임 만들기
        ###################################################################
        print(f"[2] Setting total frames to {TARGET_FRAME_COUNT} -> {frame40_path}")
        success = set_frame_count_to_target(trimmed_path, frame40_path, TARGET_FRAME_COUNT)
        if not success:
            print(f"[에러] 40프레임 변환 실패 -> 스킵: {trimmed_path}")
            continue

        ###################################################################
        # (3) 오디오 추출
        ###################################################################
        print(f"[3] Extracting audio -> {audio_path}")
        extract_audio(frame40_path, audio_path)

        ###################################################################
        # (4) 256x256 리사이즈
        ###################################################################
        print(f"[4] Resizing to {RESIZE_WIDTH}x{RESIZE_HEIGHT} -> {resized_path}")
        resize_video(frame40_path, resized_path, RESIZE_WIDTH, RESIZE_HEIGHT)

        ###################################################################
        # (5) 영상 + 오디오 합치기
        ###################################################################
        print(f"[5] Combining audio+video -> {combined_path}")
        combine_video_audio(resized_path, audio_path, combined_path)

        print(f"=== 완료: {video_name} ===\n")

    ###################################################################
    # (6) 전처리 중간 파일들 삭제
    ###################################################################
    delete_files_in_folder(OUTPUT_FOLDER)


    print("\n모든 작업이 완료되었습니다.")

###############################################################################
# 3. 엔트리 포인트
###############################################################################
if __name__ == "__main__":
    main()
