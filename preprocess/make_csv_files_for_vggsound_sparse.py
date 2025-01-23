import os
import re
import glob
import csv
import argparse
from typing import List, Optional

def _extract_category_from_filename(filename: str) -> Optional[str]:
    """
    파일 이름에서 category를 추출합니다. youtube_id는 숫자로 끝나고, category는 그 뒤에 위치합니다.
    
    :param filename: 비디오 파일 이름.
    :return: 추출된 category 또는 None.
    """
    # 파일 이름에서 기본 이름 추출
    base = os.path.basename(filename)
    
    # 정규식 패턴을 사용하여 youtube_id와 category 추출
    # 파일 이름 형식: vggsound_sparse_{youtube_id}_{category}_{split}_fixed.mp4
    

    # split -> train / test / (train|test)

    pattern = r'vggsound_sparse_(.+)_(test)_fixed\.mp4'
    match = re.match(pattern, base)
    
    if match:
        # youtube_id와 category 부분을 추출 (전체)
        full_id_category = match.group(1)
        split = match.group(2)
        
        # youtube_id는 숫자로 끝난다고 가정하고 마지막 숫자를 기준으로 구분
        # 숫자로 끝나는 youtube_id를 식별
        youtube_id_match = re.search(r'(.+_\d+)', full_id_category)
        
        if not youtube_id_match:
            print(f"youtube_id를 파일 이름에서 찾을 수 없습니다: {filename}")
            return None
        
        # youtube_id가 끝나는 위치까지 잘라서 youtube_id로 추출
        youtube_id = youtube_id_match.group(1)
        category = full_id_category[len(youtube_id)+1:]  # youtube_id 다음의 부분이 category
        
        # 확인용 출력 (옵션)
        print(f"youtube_id: {youtube_id}, category: {category}, split: {split}")
        
        return category
    else:
        print(f"파일 이름 형식이 예상과 다릅니다: {filename}")
        return None

def _list_video_files_recursively(data_dir: str) -> List[str]:
    """
    주어진 디렉토리에서 비디오 파일을 재귀적으로 탐색하여 리스트로 반환합니다.

    :param data_dir: 탐색할 디렉토리 경로.
    :return: 비디오 파일의 전체 경로 리스트.
    """
    video_extensions = ("*.avi", "*.gif", "*.mp4", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return sorted(video_files)

def save_video_paths_and_categories_to_csv(data_dir: str, output_csv: str):
    """
    주어진 디렉토리에서 비디오 파일을 찾고, 각 비디오의 경로와 카테고리를 CSV 파일로 저장합니다.

    :param data_dir: 비디오 파일이 있는 디렉토리 경로.
    :param output_csv: 저장할 CSV 파일의 경로.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
    
    video_files = _list_video_files_recursively(data_dir)
    print(f"찾은 비디오 파일 개수: {len(video_files)}")
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['video_path', 'category'])  # 헤더 작성
        
        for video_path in video_files:
            category = _extract_category_from_filename(video_path)
            if category:
                writer.writerow([video_path, category])
            else:
                print(f"카테고리를 추출할 수 없어 건너뜁니다: {video_path}")

    print(f"CSV 파일이 성공적으로 저장되었습니다: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="비디오 파일의 경로와 카테고리를 CSV로 저장하는 스크립트")
    parser.add_argument('--data_dir', type=str, default='/workspace/dataset/vggsound_sparse', help="비디오 파일이 있는 디렉토리 경로")
    parser.add_argument('--output_csv', type=str, default='/workspace/dataset/vggsound_sparse_test_video_categories.csv', help="저장할 CSV 파일 경로")
    
    args = parser.parse_args()
    
    save_video_paths_and_categories_to_csv(args.data_dir, args.output_csv)

if __name__ == "__main__":
    main()
