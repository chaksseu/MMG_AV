import os
import pandas as pd
from moviepy.editor import VideoFileClip

# ====== 사용자 설정 ======
INPUT_CSV_PATH = "/home/work/kby_hgh/MMG_01/vggsound_processing/0626_compressed_openvid_csvs/0626_processed_Openvid_test_with_audio_caption_filtered_summerize.csv"
VIDEO_DIR = "/home/work/kby_hgh/MMG_01/video_lora_training/processed_OpenVid_1920_test_full_videos_0627"  # 원본 mp4가 있는 폴더
OUTPUT_VIDEO_DIR = "/home/work/kby_hgh/MMG_01/video_lora_training/processed_OpenVid_1920_test_split_videos_0627"  # 자른 mp4 저장 폴더
OUTPUT_CSV_PATH = "/home/work/kby_hgh/MMG_01/video_lora_training/0626_processed_Openvid_split_test_with_audio_caption_filtered_summerize.csv"  # 자른 비디오 정보 CSV 저장 경로
CHUNK_DURATION = 3.2  # 초
# ==========================

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# CSV 로드
df = pd.read_csv(INPUT_CSV_PATH)
new_rows = []

for _, row in df.iterrows():
    video_id = row["id"]
    video_path = os.path.join(VIDEO_DIR, video_id)

    if not os.path.exists(video_path):
        print(f"[SKIP] Not found: {video_id}")
        continue

    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        num_parts = int(duration // CHUNK_DURATION)

        for i in range(num_parts):
            start = i * CHUNK_DURATION
            end = start + CHUNK_DURATION
            subclip = clip.subclip(start, end)

            new_id = f"{video_id.replace('.mp4', '')}_part{i+1}.mp4"
            output_path = os.path.join(OUTPUT_VIDEO_DIR, new_id)

            subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

            # 새 row 추가
            new_rows.append({
                "id": new_id,
                "new_caption": row["new_caption"],
                "split": row["split"],
                "caption": row["caption"],
                "compressed_video_caption": row["compressed_video_caption"]
            })
        clip.close()

    except Exception as e:
        print(f"[ERROR] {video_id}: {e}")

# 새 CSV 저장
new_df = pd.DataFrame(new_rows)
new_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ Splitting complete. Saved to {OUTPUT_CSV_PATH}")
