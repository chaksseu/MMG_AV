import pandas as pd
import os
import glob

# ✅ 여기만 수정하면 됩니다
input_dir = "/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs/generated_csvs/"                  # CSV들이 들어있는 폴더
output_csv_path = "/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs/llm_combined_vgg_csv_0404.csv"        # 저장할 병합 결과 경로

# 📂 해당 폴더 내 모든 CSV 파일 경로 가져오기
csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

# 🧩 모든 CSV 병합
df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# 💾 병합된 CSV 저장
merged_df.to_csv(output_csv_path, index=False)
print(f"✅ 병합 완료: {output_csv_path} ({len(merged_df)} rows)")


# import pandas as pd

# # CSV 파일 경로 설정 (파일명은 환경에 맞게 수정하세요)
# csv_file_video = '/home/work/kby_hgh/MMG_01/vggsound_processing/0401_video_llm_caption/llm_mp4_vgg_csv_0401.csv'   # 첫 번째 CSV (id, caption, split, new_caption, llm_video_caption)
# csv_file_audio = '/home/work/kby_hgh/MMG_01/vggsound_processing/0331_audio_llm_caption/audio_llm_vgg_csv_0331.csv'   # 두 번째 CSV (id, caption, split, llm_audio_caption)
# output_csv ="/home/work/kby_hgh/MMG_01/vggsound_processing/0401_LLM_VGG.csv"

# # CSV 파일 읽기
# df_video = pd.read_csv(csv_file_video)
# df_audio = pd.read_csv(csv_file_audio)

# # 두 데이터프레임을 id 열을 기준으로 병합 (audio 파일의 llm_audio_caption 컬럼만 사용)
# merged_df = pd.merge(df_video, df_audio[['id', 'llm_audio_caption']], on='id', how='inner')

# # 병합된 결과를 새로운 CSV 파일로 저장
# merged_df.to_csv(output_csv, index=False)

# print(f"병합된 CSV 파일 {output_csv}가 생성되었습니다.")
