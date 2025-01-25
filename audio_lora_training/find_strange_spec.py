import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import argparse

def is_pt_file(filename):
    return filename.endswith('.pt')

def check_file(path):
    """
    주어진 경로의 .pt 파일을 검사합니다.
    - 파일 크기가 0 byte인지 확인
    - torch.load()로 파일을 불러올 수 있는지 확인
    반환:
        (filename, error_message) 또는 None
    """
    fname = os.path.basename(path)
    
    # 파일 크기 확인
    try:
        if os.path.getsize(path) == 0:
            return (fname, "File size is 0 bytes")
    except OSError as e:
        return (fname, f"OS error: {e}")
    
    # torch.load 시도
    try:
        torch.load(path)
    except Exception as e:
        return (fname, str(e))
    
    return None  # 파일이 정상임

def gather_pt_files(spec_dir):
    """
    spec_dir 내의 모든 .pt 파일 경로를 재귀적으로 수집합니다.
    """
    pt_files = []
    for root, _, files in os.walk(spec_dir):
        for file in files:
            if is_pt_file(file):
                pt_files.append(os.path.join(root, file))
    return pt_files

def check_pt_files(spec_dir, log_file="invalid_pt_files.log", num_workers=4):
    """
    spec_dir 폴더 안의 모든 .pt 파일을 torch.load로 열어봄.
    - 실패 시(파일 손상, 잘못된 포맷 등) 파일 이름과 에러 메시지를 log_file에 기록.
    - 파일 사이즈가 0 byte인 경우도 '이상한 파일'로 간주해서 기록.
    - 병렬 처리를 통해 빠르게 검사.
    """
    pt_files = gather_pt_files(spec_dir)
    invalid_files = []
    
    print(f"총 {len(pt_files)}개의 .pt 파일을 검사합니다...")

    # 멀티프로세싱을 사용하여 파일 검사
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # tqdm을 이용한 진행 바
        futures = {executor.submit(check_file, path): path for path in pt_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="검사 중"):
            result = future.result()
            if result is not None:
                invalid_files.append(result)
    
    # 결과 로깅
    if invalid_files:
        print(f"\n[!] 총 {len(invalid_files)}개의 손상/이상 파일 발견:")
        with open(log_file, "w", encoding="utf-8") as f:
            for fname, err_msg in invalid_files:
                log_str = f"{fname}\t{err_msg}\n"
                print(log_str, end="")
                f.write(log_str)
        print(f"\n위 파일 목록이 '{log_file}'에 저장되었습니다.")
    else:
        print("\n모든 .pt 파일이 정상입니다. (이상한 파일 없음)")

def parse_args():
    parser = argparse.ArgumentParser(description="Check .pt files for corruption or invalid format.")
    parser.add_argument("--spec_dir", type=str, default="preprocessed_spec", help="Spectrogram .pt 파일들이 있는 디렉토리 경로")
    parser.add_argument("--log_file", type=str, default="invalid_pt_files.log", help="이상한 파일들을 기록할 로그 파일명")
    parser.add_argument("--num_workers", type=int, default=8, help="동시 실행할 워커 프로세스 수")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    check_pt_files(args.spec_dir, log_file=args.log_file, num_workers=args.num_workers)
