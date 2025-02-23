import os
import csv
import argparse

# Argument parser 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Generate a CSV file of matching video and audio files.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing 'audio' and 'video' folders.")
    #parser.add_argument("--output_csv", type=str, default="file_pairs.csv", help="Output CSV file path.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 기본 디렉토리 설정
    base_dir = args.base_dir
    output_csv = f"{base_dir}/file_pairs.csv"
    audio_dir = os.path.join(base_dir, "audio")
    video_dir = os.path.join(base_dir, "video")
    #output_csv = args.output_csv

    # 오디오 및 비디오 파일 목록 가져오기
    audio_files = set([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    video_files = set([f for f in os.listdir(video_dir) if f.endswith('.mp4')])

    # 파일의 베이스 이름 추출 (확장자 제외)
    audio_basenames = {os.path.splitext(f)[0] for f in audio_files}
    video_basenames = {os.path.splitext(f)[0] for f in video_files}

    # 오디오와 비디오에서 공통으로 존재하는 베이스 이름 찾기
    matching_basenames = audio_basenames.intersection(video_basenames)

    # CSV 파일 작성
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 매칭된 파일들 쌍으로 작성
        for basename in sorted(matching_basenames):
            video_path = os.path.join(video_dir, basename + ".mp4") 
            audio_path = os.path.join(audio_dir, basename + ".wav")
            writer.writerow([video_path, audio_path])

    print(f"CSV 파일이 성공적으로 생성되었습니다: {output_csv}")

if __name__ == "__main__":
    main()
