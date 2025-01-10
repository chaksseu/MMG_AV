import os
import csv
import shutil
import argparse
from collections import defaultdict
from typing import Optional

def save_video_paths_and_categories_to_csv(txt_file: str, data_dir: str, output_csv: str, split: str):
    """
    주어진 train 또는 test 텍스트 파일을 읽어 비디오 경로와 카테고리를 CSV 파일로 저장합니다.

    :param txt_file: 처리할 train.txt 또는 test.txt 파일의 경로
    :param data_dir: 비디오 파일이 있는 디렉토리 경로 (예: AVSync15/videos)
    :param output_csv: 생성할 CSV 파일의 경로
    :param split: 'train' 또는 'test'
    """
    if not os.path.isfile(txt_file):
        raise FileNotFoundError(f"TXT 파일을 찾을 수 없습니다: {txt_file}")
    
    video_entries = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 라인 형식: category/file.mp4
            if '/' not in line:
                print(f"[WARN] 잘못된 형식의 라인 (category/file.mp4 형식이어야 함): {line}")
                continue
            category, relative_path = line.split('/', 1)
            video_path = os.path.join(data_dir, category, relative_path)
            if not os.path.isfile(video_path):
                print(f"[WARN] 비디오 파일을 찾을 수 없습니다: {video_path}")
                continue
            video_entries.append((video_path, category))
    
    print(f"[INFO] 찾은 비디오 파일 개수: {len(video_entries)}")
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['video_path', 'category'])  # 헤더 작성
        for video_path, category in video_entries:
            writer.writerow([video_path, category])
    
    print(f"[INFO] CSV 파일이 성공적으로 저장되었습니다: {output_csv}")

def copy_and_rename_videos(csv_file: str, destination_dir: str, split: str):
    """
    CSV 파일을 읽고, 비디오 파일을 지정된 디렉토리로 복사하면서 이름을 변경합니다.

    :param csv_file: 비디오 경로와 카테고리가 포함된 CSV 파일의 경로
    :param destination_dir: 비디오 파일을 복사할 대상 디렉토리의 경로
    :param split: 'train' 또는 'test'
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file}")
    
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"[INFO] 대상 디렉토리를 생성했습니다: {destination_dir}")
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
                print(f"[WARN] 잘못된 행 (비어있는 정보)을 건너뜁니다: {row}")
                continue
            
            if not video_path.lower().endswith('.mp4'):
                print(f"[WARN] mp4 파일이 아니므로 건너뜁니다: {video_path}")
                continue
            
            if not os.path.isfile(video_path):
                print(f"[WARN] 비디오 파일을 찾을 수 없어 건너뜁니다: {video_path}")
                continue
            
            # 카테고리별로 번호 증가
            category_counters[category] += 1
            num = category_counters[category]

            # 새로운 파일 이름: avsync_{split}_{category}_{num}.mp4
            new_filename = f"avsync_{split}_{category}_{num}.mp4"
            destination_path = os.path.join(destination_dir, new_filename)
            
            try:
                shutil.copy2(video_path, destination_path)
                copied += 1
                print(f"[INFO] 복사 완료: {video_path} -> {destination_path}")
            except Exception as e:
                print(f"[ERROR] 파일 복사 중 오류 발생: {video_path} -> {destination_path}. 오류: {e}")
    
    print(f"\n[INFO] 총 처리된 행: {total}")
    print(f"[INFO] 성공적으로 복사된 파일: {copied}")

def main():
    parser = argparse.ArgumentParser(description="AVSync 파일 복사 및 리네이밍 스크립트")
    parser.add_argument('--data_dir', type=str, required=True, help="비디오 파일이 있는 디렉토리 경로 (예: AVSync15/videos)")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test'], help="처리할 split 선택 (train/test)")
    parser.add_argument('--train_txt', type=str, default='AVSync15/train.txt', help="train.txt 파일의 경로")
    parser.add_argument('--test_txt', type=str, default='AVSync15/test.txt', help="test.txt 파일의 경로")
    parser.add_argument('--output_csv', type=str, default=None, help="생성할 CSV 파일의 경로 (기본값: split에 따라 'AVSync15_train_video_categories.csv' 또는 'AVSync15_test_video_categories.csv')")
    parser.add_argument('--destination_dir', type=str, required=True, help="비디오 파일을 복사할 대상 디렉토리의 경로")
    
    args = parser.parse_args()
    
    split = args.split
    if split == 'train':
        txt_file = args.train_txt
    else:
        txt_file = args.test_txt
    
    if args.output_csv is None:
        # 기본 CSV 경로 설정
        output_csv = f"AVSync15_{split}_video_categories.csv"
    else:
        output_csv = args.output_csv
    
    # 1) CSV 생성
    save_video_paths_and_categories_to_csv(txt_file, args.data_dir, output_csv, split)
    
    # 2) CSV를 이용해 파일 복사 + 리네이밍
    copy_and_rename_videos(output_csv, args.destination_dir, split)

if __name__ == "__main__":
    main()
