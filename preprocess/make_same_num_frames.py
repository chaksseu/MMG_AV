import os
import subprocess
import shutil

def change_fps_in_folder(input_folder, output_folder, target_fps):
    # 출력 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 출력 폴더 생성: {output_folder}")

    # 입력 폴더 내 모든 파일 검색
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Processing {filename}...")

            # ffmpeg 명령어 구성
            command = [
                'ffmpeg',
                '-i', input_path,
                '-r', str(target_fps),
                '-y',  # 덮어쓰기 허용
                output_path
            ]

            try:
                # ffmpeg 실행
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                print(f"✅ {filename}의 FPS를 {target_fps}로 변경하여 저장 완료.")
            except subprocess.CalledProcessError as e:
                print(f"❌ {filename} 변환 중 오류 발생: {e}")

if __name__ == "__main__":
    # 입력 폴더 경로
    input_folder = input("FPS를 변경할 MP4 파일들이 있는 폴더의 전체 경로를 입력하세요: ").strip()
    
    if not os.path.isdir(input_folder):
        print("유효하지 않은 입력 폴더 경로입니다.")
        exit(1)
    
    # 출력 폴더 경로
    output_folder = input("변환된 MP4 파일을 저장할 출력 폴더의 전체 경로를 입력하세요: ").strip()
    
    # 목표 FPS 입력
    try:
        fps = float(input("목표 FPS를 입력하세요 (예: 30): ").strip())
    except ValueError:
        print("유효한 숫자를 입력하세요.")
        exit(1)
    
    change_fps_in_folder(input_folder, output_folder, fps)
    print("모든 파일의 FPS 변환이 완료되었습니다.")
