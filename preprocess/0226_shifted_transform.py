import os
import cv2
import subprocess
from pydub import AudioSegment
import glob

###############################################################################
# 0. 사용자 지정 경로 & 파라미터
###############################################################################
# AVSync15
# vggsound_sparse

ORIGINAL_FOLDER = "/workspace/0226_test_sets/vggsound_sparse_test"
OUTPUT_FOLDER   = "/workspace/0226_test_sets/0226_shifted_vggsound_sparse_test_datasets/shifted_04s_vggsound_sparse_test"

TRIM_LENGTH = 3.2
TARGET_FRAME_COUNT = 40
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 320
SHIFT_SECONDS = 0.4  # 오디오 이동할 초 단위 (양수: 뒤로 이동, 음수: 앞으로 이동)

AUDIO_FOLDER    = os.path.join(OUTPUT_FOLDER, "audio")
VIDEO_FOLDER    = os.path.join(OUTPUT_FOLDER, "video")
COMBINED_FOLDER = os.path.join(OUTPUT_FOLDER, "combined_video")

###############################################################################
# 1. 헬퍼 함수들
###############################################################################
def delete_files_in_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                print(f"파일 삭제: {item_path}")
            except Exception as e:
                print(f"파일 삭제 실패: {item_path} - {e}")
        else:
            print(f"폴더 유지: {item_path}")

def get_video_duration(video_path):
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
    trim_cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_second}",
        "-i", input_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"[영상 자르기 완료] {output_path} (시작: {start_second:.2f}s, 길이: {duration:.2f}s)")

def set_frame_count_to_target(input_path, output_path, target_frame_count, trim_length):
    new_fps = target_frame_count / trim_length
    convert_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", f"fps={new_fps:.6f}",
        "-frames:v", str(target_frame_count),
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    actual_frames = count_frames(output_path)
    if actual_frames != target_frame_count:
        print(f"[경고] {output_path}의 프레임 수가 예상과 다릅니다: {actual_frames}프레임")
        return False
    return True

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[에러] 동영상을 열 수 없습니다: {video_path}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def extract_audio(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path, format="mp4")
        audio.export(output_path, format="wav")
        print(f"[오디오 추출 완료] {output_path}")
    except Exception as e:
        print(f"[에러] 오디오 추출 실패: {input_path}, 오류: {e}")

def shift_audio(audio_path, output_path, shift_seconds):
    try:
        audio = AudioSegment.from_file(audio_path, format="wav")
        silence = AudioSegment.silent(duration=int(abs(shift_seconds) * 1000))
        if shift_seconds > 0:
            shifted_audio = silence + audio[:-int(shift_seconds * 1000)]
        else:
            shifted_audio = audio[int(abs(shift_seconds) * 1000):] + silence
        shifted_audio.export(output_path, format="wav")
        print(f"[오디오 Shift 완료] {output_path} ({shift_seconds}초 이동)")
    except Exception as e:
        print(f"[에러] 오디오 Shift 실패: {audio_path}, 오류: {e}")

def resize_video(input_path, output_path, width=256, height=256):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
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
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        output_path
    ]
    subprocess.run(command, check=True)
    print(f"[영상+오디오 합치기 완료] {output_path}")

###############################################################################
# 2. 메인 처리 함수
###############################################################################
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    os.makedirs(COMBINED_FOLDER, exist_ok=True)

    mp4_files = [
        f for f in os.listdir(ORIGINAL_FOLDER)
        if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(ORIGINAL_FOLDER, f))
    ]

    for video_name in mp4_files:
        input_path = os.path.join(ORIGINAL_FOLDER, video_name)
        total_duration = get_video_duration(input_path)
        if not total_duration or total_duration <= TRIM_LENGTH:
            print(f"[경고] 영상 길이({total_duration:.2f}s)가 {TRIM_LENGTH}s보다 짧음 -> 스킵: {video_name}")
            continue

        # 영상의 중간 구간 3.2초 선택
        middle_start = (total_duration - TRIM_LENGTH) / 2
        base_name, _ = os.path.splitext(video_name)

        # 오디오 처리 (최초 추출 -> 시프트 -> 중간 구간 자르기)
        original_audio_path = os.path.join(AUDIO_FOLDER, f"{base_name}_original.wav")
        shifted_full_audio_path = os.path.join(AUDIO_FOLDER, f"{base_name}_shifted_full.wav")
        final_audio_path = os.path.join(AUDIO_FOLDER, f"{base_name}_final.wav")
        
        # 원본 영상에서 오디오 추출 후 시프트 수행
        extract_audio(input_path, original_audio_path)
        shift_audio(original_audio_path, shifted_full_audio_path, SHIFT_SECONDS)
        try:
            shifted_audio = AudioSegment.from_file(shifted_full_audio_path, format="wav")
            start_ms = int(middle_start * 1000)
            end_ms = int((middle_start + TRIM_LENGTH) * 1000)
            trimmed_shifted_audio = shifted_audio[start_ms:end_ms]
            trimmed_shifted_audio.export(final_audio_path, format="wav")
            print(f"[Shifted 오디오 중간 구간 자르기 완료] {final_audio_path}")
        except Exception as e:
            print(f"[에러] Shifted 오디오 자르기 실패: {e}")
            continue

        # 영상 처리
        trimmed_video_path  = os.path.join(OUTPUT_FOLDER, f"{base_name}_trimmed.mp4")
        frame40_path        = os.path.join(OUTPUT_FOLDER, f"{base_name}_40frames.mp4")
        resized_path        = os.path.join(VIDEO_FOLDER, f"{base_name}.mp4")
        combined_path       = os.path.join(COMBINED_FOLDER, f"{base_name}_combined.mp4")

        # 영상 중간 구간 3.2초 잘라내기
        trim_video(input_path, trimmed_video_path, middle_start, TRIM_LENGTH)
        success = set_frame_count_to_target(trimmed_video_path, frame40_path, TARGET_FRAME_COUNT, TRIM_LENGTH)
        if not success:
            continue
        resize_video(frame40_path, resized_path, RESIZE_WIDTH, RESIZE_HEIGHT)
        combine_video_audio(resized_path, final_audio_path, combined_path)

        print(f"=== 완료: {video_name} ===\n")

    delete_files_in_folder(OUTPUT_FOLDER)



    folder_path = AUDIO_FOLDER

    # 폴더 내의 모든 .wav 파일 찾기
    wav_files = glob.glob(os.path.join(folder_path, '*.wav'))

    for file in wav_files:
        # 파일 이름이 '_final.wav'로 끝나지 않는 경우 삭제
        if not file.endswith('_final.wav'):
            os.remove(file)
            print(f"삭제됨: {file}")

    def rename_wav_files(directory):
        for filename in os.listdir(directory):
            if filename.endswith("_final.wav"):
                old_path = os.path.join(directory, filename)
                new_filename = filename.replace("_final.wav", ".wav")
                new_path = os.path.join(directory, new_filename)

                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

    # 사용할 폴더 경로 지정
    rename_wav_files(folder_path)




    print("\n모든 작업이 완료되었습니다.")

###############################################################################
# 3. 엔트리 포인트
###############################################################################
if __name__ == "__main__":
    main()
