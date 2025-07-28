import os
import re
import pandas as pd
from collections import defaultdict



model_name_list = ["/home/work/kby_hgh/0606_FILTERED_FULL_MMG_RC_LINEAR_T_step", "/home/work/kby_hgh/0606_FILTERED_FULL_MMG_NAIVE_DISTILL_step", "/home/work/kby_hgh/0610_FILTERED_FULL_MMG_OURS_step", "/home/work/kby_hgh/0608_ALL_FULL_MMG_OURS_step"]
checkpoint_dir_list = ["2015", "4031", "6047", "8063", "10079", "12095", "14111", "16127", "18143", "20159", "22175", "24191", "26207", "28223", "30239", "32255", "34271", "36287", "38303", "40319", "42335", "44351", "46367", "48383", "50399", "52415", "54431", "56447", "58463", "60479"]
model_name_list=[0]
checkpoint_dir_list=[0]
for model_name in model_name_list:
    for checkpoint_num in checkpoint_dir_list:


        # folder_path = f"{model_name}_{checkpoint_num}_openvid/video"
        folder_path = "/home/work/kby_hgh/0603_video_teacher_FULL_openvid_inference/checkpoint-step-81920"

        if not os.path.isdir(folder_path):
            print(f"{folder_path} does not exist. Skipping")
            continue


        filtered_csv_path = "/home/work/kby_hgh/MMG_01/video_lora_training/0602_processed_Openvid_test_with_audio_caption_filtered.csv"
        filtered_df = pd.read_csv(filtered_csv_path)




        # 정규표현식 패턴: batch 뒤 숫자 추출
        batch_number_pattern = re.compile(r"_batch_(\d+)")


        # 1. 폴더에서 기준 id 리스트 생성
        folder_ids = set()
        id_to_files = defaultdict(list)

        for filename in os.listdir(folder_path):
            if "_batch_" in filename:
                base_id = filename.split("_batch_")[0] + ".mp4"
                folder_ids.add(base_id)
                id_to_files[base_id].append(filename)
            else:
                base_id = filename
                folder_ids.add(base_id)
                id_to_files[base_id].append(filename)

        # # 2. CSV에서 id 리스트 불러오기
        # csv_ids = set(df['id'].tolist())

        # # 3. CSV에만 있고 폴더에는 없는 것 제거
        # to_remove_from_csv = csv_ids - folder_ids
        # filtered_df = df[~df['id'].isin(to_remove_from_csv)]

        # # 4. 필터링된 CSV 저장
        # filtered_csv_path = csv_path.replace(".csv", "_filtered.csv")
        # filtered_df.to_csv(filtered_csv_path, index=False)
        # print(f"[CSV 정리] 총 {len(to_remove_from_csv)}개 항목 제거됨.")

        # 5. CSV에 없는 ID 기준으로 폴더 내 파일 제거 + 동일 ID 처리
        filtered_ids = set(filtered_df['id'].tolist())
        deleted_count = 0

        for base_id, files in id_to_files.items():
            if base_id not in filtered_ids:
                # CSV에 없는 ID: 전부 삭제
                for f in files:
                    file_path = os.path.join(folder_path, f)
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"[삭제됨] {file_path}")
            elif len(files) > 1:
                # 여러 개 있을 경우 가장 작은 batch 번호만 남기기
                def extract_batch_num(filename):
                    match = batch_number_pattern.search(filename)
                    return int(match.group(1)) if match else float('inf')  # 매치 안 되면 마지막에 정렬

                files_sorted = sorted(files, key=extract_batch_num)
                to_keep = files_sorted[0]
                for f in files_sorted[1:]:
                    file_path = os.path.join(folder_path, f)
                    os.remove(file_path)
                    deleted_count += 1
                    # print(f"[중복 제거] {file_path} (남김: {to_keep})")

        print(f"[폴더 정리] 총 {deleted_count}개 파일 삭제됨.")