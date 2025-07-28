import pandas as pd
import os

def merge_csv_files(file_paths, output_path):
    # 각 파일 읽어서 DataFrame 리스트로 저장
    dfs = [pd.read_csv(path) for path in file_paths]
    
    # 모두 concat
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 저장
    merged_df.to_csv(output_path, index=False)
    print(f"✅ 통합된 CSV가 저장되었습니다: {output_path}")

# 예시 사용법
if __name__ == "__main__":
    # 병합할 CSV 파일 경로 리스트
    csv_paths = [
        "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_0.csv",
        "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_1.csv",
        "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_2.csv",
        "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_3.csv",
        "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_4.csv",
        "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_5.csv",
        "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_6.csv",
        "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/splited_output/0623_processed_Openvid_train_with_audio_caption_summerize_7.csv",
    ]

    # 출력 파일 경로
    output_csv = "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/0626_processed_Openvid_train_with_audio_caption_summerize_merge.csv"
    
    merge_csv_files(csv_paths, output_csv)