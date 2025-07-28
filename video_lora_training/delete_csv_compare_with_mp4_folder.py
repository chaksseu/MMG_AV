import pandas as pd
import os

# 설정값
csv_path = "/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_test_with_audio_caption.csv"  # CSV 파일 경로
folder_path = "/home/work/kby_hgh/MMG_01/video_lora_training/processed_OpenVid_2000_test_videos_42"  # 비교할 폴더 경로
file_extension = ".mp4"  # 예: '1234.wav'라면 '.wav'

# CSV 로드
df = pd.read_csv(csv_path)

# 폴더 내 존재하는 파일들의 ID 집합 생성 (확장자 제거)
existing_ids = {
    filename
    for filename in os.listdir(folder_path)
    if filename.endswith(file_extension)
}
print(existing_ids)
# CSV에서 해당 ID만 필터링
filtered_df = df[df["id"].astype(str).isin(existing_ids)]

# 결과 저장
filtered_df.to_csv("filtered.csv", index=False)

# 개수 출력
print(f"Original CSV rows: {len(df)}")
print(f"Filtered CSV rows: {len(filtered_df)}")
print(f"Removed rows: {len(df) - len(filtered_df)}")
