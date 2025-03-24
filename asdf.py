import pandas as pd
import os

# 경로 설정
csv_path = "/workspace/data/MMG_TA_dataset_audiocaps_wavcaps/MMG_TA_dataset_preprocessed.csv"
spec_dir = "/workspace/data/MMG_TA_dataset_audiocaps_wavcaps_spec_0320"
output_csv_path = "/workspace/data/MMG_TA_dataset_audiocaps_wavcaps/MMG_TA_dataset_filtered_0321.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 존재하는 .pt 파일명 set 생성 (확장자 제거)
existing_ids = {
    os.path.splitext(fname)[0]
    for fname in os.listdir(spec_dir)
    if fname.endswith(".pt")
}

# 필터링: id 열 값이 existing_ids에 있는 행만 선택
filtered_df = df[df["id"].astype(str).isin(existing_ids)]

# 결과 저장
filtered_df.to_csv(output_csv_path, index=False)

print(f"Filtered CSV saved to: {output_csv_path}")
