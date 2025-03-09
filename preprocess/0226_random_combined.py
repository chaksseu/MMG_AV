import os
import random
import subprocess
import shutil

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

def combine_random_pairs(video_dir, audio_dir, output_dir):
    # 출력 폴더 내에 서브폴더 생성: video, audio, combined
    video_out = os.path.join(output_dir, "video")
    audio_out = os.path.join(output_dir, "audio")
    combined_out = os.path.join(output_dir, "combined")
    os.makedirs(video_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(combined_out, exist_ok=True)
    
    # video와 audio 디렉토리 내의 파일 리스트 가져오기
    video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    audio_files = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
    
    # 리스트를 랜덤하게 섞기
    random.shuffle(video_files)
    random.shuffle(audio_files)
    
    # 가능한 쌍의 개수는 두 리스트 중 짧은 리스트의 길이로 결정
    pair_count = min(len(video_files), len(audio_files))
    
    for i in range(pair_count):
        # 원본 파일 경로
        src_video = os.path.join(video_dir, video_files[i])
        src_audio = os.path.join(audio_dir, audio_files[i])
        
        # 파일 확장자 추출 (ex: .mp4, .wav 등)
        video_ext = os.path.splitext(video_files[i])[1]
        audio_ext = os.path.splitext(audio_files[i])[1]
        
        # 새로운 파일명 (숫자로 지정)
        new_video_name = f"{i+1}{video_ext}"
        new_audio_name = f"{i+1}{audio_ext}"
        new_combined_name = f"{i+1}.mp4"  # 합쳐진 파일은 mp4로 저장
        
        # 출력 경로
        dest_video = os.path.join(video_out, new_video_name)
        dest_audio = os.path.join(audio_out, new_audio_name)
        dest_combined = os.path.join(combined_out, new_combined_name)
        
        # 원본 파일을 각각 복사
        shutil.copy(src_video, dest_video)
        shutil.copy(src_audio, dest_audio)
        
        # 복사된 파일을 사용하여 영상+오디오 합치기
        combine_video_audio(dest_video, dest_audio, dest_combined)

if __name__ == "__main__":
    # 각 디렉토리 경로 설정 (필요에 따라 수정)
    video_directory = "/workspace/processed_unseen_AVSync15/video"
    audio_directory = "/workspace/processed_unseen_AVSync15/audio"
    output_directory = "/workspace/0226_test_sets/0226_shifted_unseen_AVSync15_datasets/random_AVSync15_test"
    
    combine_random_pairs(video_directory, audio_directory, output_directory)
