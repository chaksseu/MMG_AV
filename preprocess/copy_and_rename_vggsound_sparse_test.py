import os
import csv
import shutil
import argparse
from collections import defaultdict

def copy_and_rename_videos(csv_file: str, destination_dir: str):
    """
    CSV 파일을 읽고, mp4 파일들을 새로운 디렉토리로 복사하면서 지정된 형식으로 이름을 변경합니다.
    
    :param csv_file: 비디오 경로와 카테고리가 포함된 CSV 파일의 경로.
    :param destination_dir: 비디오 파일들을 복사할 대상 디렉토리의 경로.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file}")
    
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"대상 디렉토리를 생성했습니다: {destination_dir}")
    elif not os.path.isdir(destination_dir):
        raise NotADirectoryError(f"대상 경로가 디렉토리가 아닙니다: {destination_dir}")
    
    # 카테고리별 번호를 추적하기 위한 딕셔너리
    category_counters = defaultdict(int)
    
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total = 0
        copied = 0
        for row in reader:
            total += 1
            video_path = row.get('video_path', '').strip()
            category = row.get('category', '').strip()
            
            if not video_path or not category:
                print(f"잘못된 행을 건너뜁니다: {row}")
                continue
            
            if not video_path.lower().endswith('.mp4'):
                print(f"mp4 파일이 아니므로 건너뜁니다: {video_path}")
                continue
            
            if not os.path.isfile(video_path):
                print(f"비디오 파일을 찾을 수 없어 건너뜁니다: {video_path}")
                continue
            
            # 카테고리별로 번호 증가
            category_counters[category] += 1
            num = category_counters[category]
            
            # 새로운 파일 이름 생성
            new_filename = f"vggsound_sparse_test_{category}_{num}.mp4"
            destination_path = os.path.join(destination_dir, new_filename)
            
            try:
                shutil.copy2(video_path, destination_path)
                copied += 1
                print(f"복사 완료: {video_path} -> {destination_path}")
            except Exception as e:
                print(f"파일 복사 중 오류 발생: {video_path} -> {destination_path}. 오류: {e}")
    
    print(f"\n총 처리된 행: {total}")
    print(f"성공적으로 복사된 파일: {copied}")

def main():
    parser = argparse.ArgumentParser(description="CSV 파일을 기반으로 mp4 파일을 복사하고 이름을 변경하는 스크립트")
    parser.add_argument('--csv_file', type=str, default="/workspace/dataset/vggsound_sparse_test_video_categories.csv", help="비디오 경로와 카테고리가 포함된 CSV 파일의 경로")
    parser.add_argument('--destination_dir', type=str, default="/workspace/dataset/vggsound_sparse_test", help="비디오 파일들을 복사할 대상 디렉토리의 경로")
    
    args = parser.parse_args()
    
    copy_and_rename_videos(args.csv_file, args.destination_dir)

if __name__ == "__main__":
    main()
