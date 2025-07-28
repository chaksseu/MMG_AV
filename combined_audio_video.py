
import os
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip

def save_video_with_audio(video_path, audio_path, savedir, base_filename, fps=12.5):
    os.makedirs(savedir, exist_ok=True)
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        video_with_audio = video_clip.set_audio(audio_clip)
        savepath = os.path.join(savedir, f"{base_filename}.mp4")
        video_with_audio.write_videofile(
            savepath,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(savedir, f"{base_filename}_temp_audio.m4a"),
            remove_temp=True,
            fps=fps,
            verbose=False,
            logger=None
        )
        print(f"[✓] Saved: {savepath}")
    except Exception as e:
        print(f"[✗] Skipped {base_filename} due to error: {e}")

def combine_audio_video_folder(video_dir, audio_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    video_files = {os.path.splitext(f)[0]: os.path.join(video_dir, f)
                   for f in os.listdir(video_dir) if f.endswith('.mp4')}
    audio_files = {os.path.splitext(f)[0]: os.path.join(audio_dir, f)
                   for f in os.listdir(audio_dir) if f.endswith('.wav')}
    common_keys = set(video_files.keys()) & set(audio_files.keys())
    print(f"Found {len(common_keys)} matching video/audio pairs.")
    for key in sorted(common_keys):
        video_path = video_files[key]
        audio_path = audio_files[key]
        save_video_with_audio(video_path, audio_path, save_dir, key)

# 0717_ALL_FULL_MMG_OURS_step_8063_audiocaps 0717_ALL_FULL_MMG_OURS_step_8063_openvid 0717_ALL_FULL_MMG_OURS_step_8063_vggsound_sparse
# 0710_FILTERED_FULL_MMG_OURS_step_12095_audiocaps 0710_FILTERED_FULL_MMG_OURS_step_12095_openvid 0710_FILTERED_FULL_MMG_OURS_step_12095_vggsound_sparse
# 0717_FILTERED_FULL_MMG_NAIVE_DISTILL_step_16127_audiocaps 0717_FILTERED_FULL_MMG_NAIVE_DISTILL_step_16127_openvid 0717_FILTERED_FULL_MMG_NAIVE_DISTILL_step_16127_vggsound_sparse
# 0710_FILTERED_FULL_MMG_RC_LENEAR_T_1_2_1_2_step_16127_audiocaps 0710_FILTERED_FULL_MMG_RC_LENEAR_T_1_2_1_2_step_16127_openvid 0710_FILTERED_FULL_MMG_RC_LENEAR_T_1_2_1_2_step_16127_vggsound_sparse


def main():
    parser = argparse.ArgumentParser(description="Combine matching .mp4 and .wav files into videos with audio.")
    parser.add_argument('--video_dir', default='/home/work/kby_hgh/0717_FILTERED_FULL_MMG_NAIVE_DISTILL_step_16127_audiocaps/video', help='Directory containing .mp4 video files')
    parser.add_argument('--audio_dir', default='/home/work/kby_hgh/0717_FILTERED_FULL_MMG_NAIVE_DISTILL_step_16127_audiocaps/audio', help='Directory containing .wav audio files')
    parser.add_argument('--save_dir', default='/home/work/kby_hgh/0717_FILTERED_FULL_MMG_NAIVE_DISTILL_step_16127_audiocaps/combined', help='Directory to save the combined output videos')
    args = parser.parse_args()

    combine_audio_video_folder(args.video_dir, args.audio_dir, args.save_dir)

if __name__ == "__main__":
    main()
