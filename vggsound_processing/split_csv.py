import pandas as pd
import os

input_csv_path = "/home/work/kby_hgh/MMG_01/vggsound_processing/0401_LLM_VGG.csv"   # 원본 CSV 경로
output_dir = "/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs"                 # 나눠서 저장할 폴더
n_splits = 8                        # 나눌 파일 개수

df = pd.read_csv(input_csv_path)

os.makedirs(output_dir, exist_ok=True)

total_rows = len(df)
split_size = (total_rows + n_splits - 1) // n_splits  # 올림 나누기

for i in range(n_splits):
    start_idx = i * split_size
    end_idx = min(start_idx + split_size, total_rows)
    split_df = df.iloc[start_idx:end_idx]

    output_path = os.path.join(output_dir, f'split_{i}.csv')
    split_df.to_csv(output_path, index=False)
    print(f"✅ 저장 완료: {output_path} ({len(split_df)} rows)")
