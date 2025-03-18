import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import logging
import sys
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """로깅 설정을 초기화합니다."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('audio_processing.log', encoding='utf-8')
        ]
    )

def check_ffmpeg_installed():
    """FFmpeg가 시스템에 설치되어 있는지 확인합니다."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("FFmpeg가 설치되어 있지 않거나 PATH에 포함되어 있지 않습니다.")
        sys.exit(1)

def process_file(input_path, output_path, retries=2):
    """
    FFmpeg를 사용하여 mp4 파일에서 오디오를 추출하고,
    모노로 변환하며 샘플링 레이트를 16000으로 설정하여 wav 파일로 저장합니다.
    """
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-ac', '1',        # 모노로 변환
        '-ar', '16000',    # 샘플링 레이트 16000Hz
        '-vn',             # 비디오 스트림 무시 (오디오만 추출)
        '-y',              # 출력 파일 덮어쓰기
        str(output_path)
    ]
    attempt = 0
    while attempt <= retries:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
            return True, str(input_path), None
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else 'Unknown error'
            logging.warning(f"FFmpeg 오류 (시도 {attempt+1}/{retries+1}): {input_path} - {error_msg}")
            attempt += 1
        except subprocess.TimeoutExpired:
            logging.warning(f"FFmpeg 타임아웃 (시도 {attempt+1}/{retries+1}): {input_path}")
            attempt += 1
    return False, str(input_path), f"Failed after {retries+1} attempts"

def main(input_folder, output_folder, num_workers, overwrite):
    # 출력 폴더가 존재하지 않으면 생성
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 입력 폴더 내 모든 mp4 파일 목록을 재귀적으로 검색
    mp4_files = list(Path(input_folder).rglob('*.mp4'))
    total_files = len(mp4_files)

    if total_files == 0:
        logging.info("입력 폴더에 mp4 파일이 없습니다.")
        return

    logging.info(f"총 {total_files}개의 mp4 파일을 처리합니다.")

    success_count = 0
    failed_files = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {}
        for file_path in mp4_files:
            # 입력 파일의 상대 경로를 기준으로 출력 경로 생성하고 확장자를 .wav로 변경
            relative_path = file_path.relative_to(input_folder).with_suffix('.wav')
            output_path = output_folder / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not overwrite:
                logging.info(f"이미 존재하여 건너뜀: {output_path}")
                total_files -= 1
                continue
            future = executor.submit(process_file, file_path, output_path)
            future_to_file[future] = file_path

        # 진행 상황 표시
        with tqdm(total=total_files, desc="처리 중", unit="파일") as pbar:
            for future in as_completed(future_to_file):
                success, filename, error = future.result()
                if success:
                    success_count += 1
                else:
                    failed_files.append((filename, error))
                    logging.error(f"처리에 실패한 파일: {filename} - {error}")
                pbar.update(1)

    # 최종 요약
    logging.info(f"처리 완료: {success_count}/{total_files} 파일 성공")
    if failed_files:
        logging.warning(f"{len(failed_files)}개의 파일 처리에 실패했습니다. 상세 내용은 로그를 확인하세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mp4 파일의 오디오를 추출하여 모노, 16000Hz wav 파일로 새 폴더에 저장합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_folder', '-i', type=str, default="/workspace/data/vggsound_train",
        help='입력 mp4 파일이 있는 폴더 경로'
    )
    parser.add_argument(
        '--output_folder', '-o', type=str, default="/workspace/data/preprocessed_VGGSound_train_audio_0310",
        help='처리된 wav 파일을 저장할 새 폴더 경로'
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=16,
        help='병렬로 처리할 워커 수'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='출력 파일을 덮어씁니다.'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='로깅 레벨'
    )

    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    check_ffmpeg_installed()
    main(args.input_folder, args.output_folder, args.workers, args.overwrite)
