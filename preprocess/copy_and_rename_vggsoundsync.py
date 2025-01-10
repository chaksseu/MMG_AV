import os
import re
import glob
import csv
import shutil
import argparse
from collections import defaultdict
from typing import Dict, Tuple

def load_csv_to_dict(csv_path: str) -> Dict[Tuple[str, int], str]:
    """
    vggsoundsync.csv 파일을 읽어 (youtube_id, start_seconds) -> label 형태로 반환합니다.
    CSV 예시:
    YouTube_ID,start_seconds,label,,,
    -0jeONf82dE,21,horse neighing,,,
    ...
    """
    # (youtube_id, start_seconds)를 키로, label을 값으로
    mapping = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # CSV 컬럼 이름 확인 (YouTube_ID, start_seconds, label)
            yt_id = row.get('YouTube_ID', '').strip()
            start_str = row.get('start_seconds', '').strip()
            label = row.get('label', '').strip()

            if not yt_id or not start_str or not label:
                continue

            try:
                start_sec = int(start_str)
            except ValueError:
                # start_seconds가 정수가 아닐 경우 무시
                continue

            mapping[(yt_id, start_sec)] = label

    return mapping

def _parse_filename(filename: str) -> Tuple[str, int]:
    """
    vggsoundsync_F9bJVVYgFl4_73_fixed.mp4 와 같은 파일명에서
    (youtube_id, start_seconds)를 파싱해서 반환합니다.

    예:
    vggsoundsync_{youtube_id}_{start_seconds}_fixed.mp4
    -> (F9bJVVYgFl4, 73)
    """
    base = os.path.basename(filename)
    # 정규식으로 추출
    # 그룹1: [A-Za-z0-9_-]+ (YouTube ID)
    # 그룹2: [0-9]+ (start_seconds)
    # 예) vggsoundsync_ABC123_-xyz_99_fixed.mp4 를 고려한다면?
    #     "vggsoundsync_" + (여러 문자) + "_" + (숫자) + "_fixed.mp4"
    pattern = r'^vggsoundsync_([A-Za-z0-9_\-\+]+)_(\d+)_fixed\.mp4$'
    match = re.match(pattern, base)
    if not match:
        return "", -1  # 파싱 불가 시

    youtube_id = match.group(1)
    start_seconds_str = match.group(2)

    try:
        start_seconds = int(start_seconds_str)
    except ValueError:
        start_seconds = -1

    return youtube_id, start_seconds

def rename_vggsoundsync_videos(data_dir: str,
                               csv_path: str,
                               destination_dir: str,
                               split: str = 'train'):
    """
    data_dir에 있는 vggsoundsync_*.mp4 파일들을 CSV(YouTube_ID, start_seconds, label) 정보와 매칭하여,
    vggsoundsync_{split}_{label}_{카운팅}.mp4 (혹은 원하는 규칙) 으로 복사합니다.
    """
    # 1) CSV 로드
    id_sec_to_label = load_csv_to_dict(csv_path)
    print(f"[INFO] CSV로부터 {len(id_sec_to_label)}개의 (YouTube_ID, start_seconds) -> label 정보를 로드했습니다.")

    # 2) 복사 대상 폴더 생성
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"[INFO] 대상 디렉토리 생성: {destination_dir}")

    # label별로 몇 개나 복사했는지 카운트
    label_counters = defaultdict(int)

    # 3) data_dir 아래의 mp4 파일들 탐색
    video_files = glob.glob(os.path.join(data_dir, "*.mp4"))
    total = 0
    copied = 0

    for video_path in video_files:
        total += 1
        base = os.path.basename(video_path)

        # 파일명에서 (youtube_id, start_seconds) 파싱
        youtube_id, start_seconds = _parse_filename(base)
        if youtube_id == "" or start_seconds < 0:
            print(f"[WARN] 파일 이름이 형식에 맞지 않아 건너뜀: {base}")
            continue

        # CSV 매핑에서 label 찾기
        key = (youtube_id, start_seconds)
        if key not in id_sec_to_label:
            print(f"[WARN] CSV에 없는 (YouTube_ID, start_seconds) 조합 -> 건너뜀: {key}")
            continue

        label = id_sec_to_label[key]

        # label 안에 공백, 특수문자 등이 있을 수 있으므로 파일명 안전하게 변환(예: 공백->언더바)
        safe_label = re.sub(r'[^A-Za-z0-9가-힣_]+', '_', label)  # 필요에 따라 수정
        label_counters[safe_label] += 1
        count = label_counters[safe_label]

        # 새 파일 이름 정의 (원하시는 규칙으로 바꿔도 됨)
        #new_filename = f"vggsoundsync_{split}_{safe_label}_{count}.mp4"
        new_filename = f"vggsoundsync_{safe_label}_{count}.mp4"
        destination_path = os.path.join(destination_dir, new_filename)

        # 파일 복사
        try:
            shutil.copy2(video_path, destination_path)
            copied += 1
            print(f"[INFO] 복사 완료: {video_path} -> {destination_path}")
        except Exception as e:
            print(f"[ERROR] 파일 복사 실패: {video_path} -> {destination_path}, 오류: {e}")

    print(f"\n[INFO] 총 파일 개수: {total}, 성공적으로 복사된 파일: {copied}")

def main():
    parser = argparse.ArgumentParser(description="vggsoundsync 파일 이름 재생성 및 복사 스크립트")
    parser.add_argument('--data_dir', type=str, default='vggsoundsync',
                        help="vggsoundsync_*.mp4가 존재하는 디렉토리 경로")
    parser.add_argument('--csv_path', type=str, default='vggsoundsync.csv',
                        help="vggsoundsync.csv 파일 경로")
    parser.add_argument('--destination_dir', type=str, default='vggsoundsync_renamed',
                        help="새로 복사할 디렉토리")
    parser.add_argument('--split', type=str, default='train',
                        help="train, test 등 구분자용 문자열 (파일 이름에 포함)")
    args = parser.parse_args()

    rename_vggsoundsync_videos(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        destination_dir=args.destination_dir,
        split=args.split
    )

if __name__ == "__main__":
    main()
