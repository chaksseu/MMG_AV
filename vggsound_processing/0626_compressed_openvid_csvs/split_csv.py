import pandas as pd
import os

def split_csv_by_rows(input_path, output_dir, num_splits, base_filename='split'):
    # CSV 불러오기
    df = pd.read_csv(input_path)
    total_rows = len(df)
    rows_per_split = total_rows // num_splits
    remainder = total_rows % num_splits

    os.makedirs(output_dir, exist_ok=True)

    start = 0
    for i in range(num_splits):
        # 마지막 분할에는 나머지까지 포함
        end = start + rows_per_split + (1 if i < remainder else 0)
        split_df = df.iloc[start:end]
        output_path = os.path.join(output_dir, f"{base_filename}_{i+1}.csv")
        split_df.to_csv(output_path, index=False)
        print(f"Saved {output_path} with {len(split_df)} rows.")
        start = end


split_csv_by_rows(
    input_path="/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_train_with_audio_caption.csv",
    output_dir="/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs",
    num_splits=8,
    base_filename='original'
)