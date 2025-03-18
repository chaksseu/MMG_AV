import pandas as pd

# 두 개의 CSV 파일 읽기
csv1 = pd.read_csv("final_audiocaps_split.csv")
csv2 = pd.read_csv("processed_WavCaps_split.csv")

# 두 CSV 파일 병합 (중복 행 포함)
concatenated_csv = pd.concat([csv1, csv2])

# 중복 행의 개수 계산
duplicates_count = len(concatenated_csv) - len(concatenated_csv.drop_duplicates())

# 병합 후 중복 제거
merged_csv = concatenated_csv.drop_duplicates().reset_index(drop=True)

# 병합된 CSV 파일 저장
merged_csv.to_csv("merged_file.csv", index=False)

print(f"병합된 파일이 'merged_file.csv'로 저장되었습니다.")
print(f"중복된 행의 개수는 {duplicates_count}개입니다.")
