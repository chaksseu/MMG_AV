import os
import csv
import glob

# 폴더 경로 설정 (mapping.csv와 mp4 파일들이 있는 폴더)
folder_path = "/workspace/data/trainvideo_10"  # 실제 폴더 경로로 변경하세요

# mapping CSV 파일 경로
mapping_csv_path = "/workspace/vggsound_processing/vggsound_modified.csv"
new_csv = "./New_VGGSound.csv"

# mapping CSV 읽기
mapping_dict = {}
with open(mapping_csv_path, mode='r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # modified_caption을 key로, 원본 caption을 value로 저장
        mapping_dict[row['modified_caption']] = row['caption']

output_rows = []
mp4_pattern = os.path.join(folder_path, "*.mp4")
mp4_files = glob.glob(mp4_pattern)

for mp4_path in mp4_files:
    base_name = os.path.basename(mp4_path)  # 예: VGG_playingtennis_12345.mp4
    try:
        modified_caption = base_name.rsplit('_', 1)[0]  # 마지막 "_" 기준으로 split 후 앞부분 추출
    except ValueError:
        print(f"파일명 형식 오류 (언더바('_')가 없음): {base_name}")
        continue

    # mapping CSV에서 해당 modified_caption에 대응하는 원본 caption 찾기
    caption = mapping_dict.get(modified_caption)
    if caption is None:
        print(f"경고: mapping에 없는 modified_caption '{modified_caption}' (파일명: {base_name})")
        caption = modified_caption  # 그대로 사용하거나 continue로 건너뛸 수 있음

    # id 값은 확장자 제거한 파일명
    id_val = base_name[:-4]

    # split 값은 "train"으로 지정
    output_rows.append({
        "id": id_val,
        "caption": caption,
        "split": "train"
    })

# 새 CSV 파일 생성
output_csv_path = os.path.join(new_csv)
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["id", "caption", "split"])
    writer.writeheader()
    writer.writerows(output_rows)

print(f"새 CSV 파일이 생성되었습니다: {output_csv_path}")
