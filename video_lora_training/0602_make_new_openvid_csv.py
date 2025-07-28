import pandas as pd

# 원본 CSV 파일 경로
original_csv_path = "/home/work/kby_hgh/splitted_openvid_csvs/0506_processed_Openvid_train_with_audio_caption_merged.csv"

# CSV 로드
df = pd.read_csv(original_csv_path)

# 랜덤 샘플링 (2000개)
sampled_df = df.sample(n=2050, random_state=42)

# split 열을 'test'로 설정
sampled_df['split'] = 'test'

# 나머지 데이터
remaining_df = df.drop(sampled_df.index)

# 새로운 CSV 파일 저장
sampled_df.to_csv("/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_test_with_audio_caption.csv", index=False)
remaining_df.to_csv("/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_train_with_audio_caption.csv", index=False)

# 각 CSV의 행 개수 출력
print(f"Original CSV rows: {len(df)}")
print(f"Sampled test CSV rows: {len(sampled_df)}")
print(f"Remaining CSV rows: {len(remaining_df)}")
