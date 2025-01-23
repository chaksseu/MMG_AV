import os
import re
import glob
import csv
import shutil
import argparse
from collections import defaultdict
from typing import List, Optional

def _extract_category_from_filename(filename: str, split: str) -> Optional[str]:
    """
    파일 이름에서 category를 추출합니다.
    파일 이름 형식: vggsound_sparse_{youtube_id}_{category}_{split}_fixed.mp4
    split: train 또는 test
    """
    base = os.path.basename(filename)

    # 사용자로부터 받은 split(train 또는 test)에 맞는 정규식 패턴 생성
    pattern = rf'vggsound_sparse_(.+)_{split}_fixed\.mp4'
    match = re.match(pattern, base)
    if match:
        # youtube_id와 category가 섞여 있는 부분(전체)
        full_id_category = match.group(1)
        
        #print(full_id_category)
        # youtube_id가 숫자로 끝난다고 가정
        # 예: vggsound_sparse_abc123_새소리_train_fixed.mp4
        #     -> full_id_category = "abc123_새소리"
        # 여기서 youtube_id = "abc123", category = "새소리"
        #youtube_id_match = re.search(r'(.+_\d+)$', full_id_category)
        youtube_id_match = re.search(r'(.+_\d+)', full_id_category)
        if not youtube_id_match:
            # youtube_id가 숫자로 끝나지 않는 경우
            print(f"[WARN] youtube_id를 파일 이름에서 찾을 수 없습니다: {filename}")
            return None

        youtube_id = youtube_id_match.group(1)
        # youtube_id 앞 부분(underbar 포함)만큼 잘라낸 후 남은 부분이 category
        # 예) full_id_category = "abc123_새소리"
        #     youtube_id      = "abc123"
        #     len(youtube_id) = 6
        #     full_id_category[len(youtube_id)+1:] => "새소리"
        category = full_id_category[len(youtube_id) + 1:]

        print(f"[INFO] youtube_id: {youtube_id}, category: {category}, split: {split}")
        return category
    else:
        print(f"[WARN] 파일 이름 형식이 예상과 다릅니다(또는 split 불일치): {filename}")
        return None

def _list_video_files_recursively(data_dir: str) -> List[str]:
    """
    주어진 디렉토리에서 비디오 파일(mp4 등)을 재귀적으로 탐색하여 리스트로 반환.
    """
    video_extensions = ("*.avi", "*.gif", "*.mp4", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return sorted(video_files)

def save_video_paths_and_categories_to_csv(data_dir: str, output_csv: str, split: str):
    """
    data_dir 아래에서 지정된 split(train/test)에 해당하는 mp4 파일을 찾아
    [video_path, category] 정보를 CSV에 기록합니다.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
    
    video_files = _list_video_files_recursively(data_dir)
    print(f"[INFO] 찾은 비디오 파일 개수: {len(video_files)}")

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['video_path', 'category'])  # 헤더

        for video_path in video_files:
            # mp4 파일만 대상으로 삼는다.
            if not video_path.lower().endswith('.mp4'):
                continue

            category = _extract_category_from_filename(video_path, split)
            if category:
                writer.writerow([video_path, category])
            else:
                # split 불일치나 형식 불일치인 경우
                pass

    print(f"[INFO] CSV 파일이 성공적으로 저장되었습니다: {output_csv}")

def copy_and_rename_videos(csv_file: str, destination_dir: str, split: str):
    """
    CSV 파일을 읽고, mp4 파일들을 새로운 디렉토리로 복사하며,
    vggsound_sparse_{split}_{category}_{num}.mp4 의 형식으로 이름을 붙입니다.
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
                print(f"[WARN] 잘못된 행(비어있는 정보)을 건너뜁니다: {row}")
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

            # 새로운 파일 이름: vggsound_sparse_{split}_{category}_{번호}.mp4
            new_filename = f"vggsound_sparse_{split}_{category}_{num}.mp4"
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
    parser = argparse.ArgumentParser(description="VGGSound Sparse 파일 처리 스크립트 (CSV 생성 + 파일 복사 및 리네이밍)")
    parser.add_argument('--data_dir', type=str, default='/workspace/dataset/vggsound_sparse',
                        help="비디오가 존재하는 디렉토리 경로")
    parser.add_argument('--output_csv', type=str, default='/workspace/dataset/vggsound_sparse_video_categories.csv',
                        help="생성할 CSV 파일 경로")
    parser.add_argument('--destination_dir', type=str, default='/workspace/dataset/vggsound_sparse_train',
                        help="복사할 대상 디렉토리 경로")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help="처리할 split 선택 (train/test)")
    args = parser.parse_args()

    # 1) CSV 생성
    save_video_paths_and_categories_to_csv(args.data_dir, args.output_csv, args.split)

    # 2) CSV를 이용해 파일 복사 + 리네이밍
    copy_and_rename_videos(args.output_csv, args.destination_dir, args.split)

if __name__ == "__main__":
    main()
