from moviepy.editor import VideoFileClip, AudioFileClip

def merge_audio_video(video_path, audio_path, output_path):
    # 비디오 파일 불러오기
    video_clip = VideoFileClip(video_path)
    
    # 오디오 파일 불러오기
    audio_clip = AudioFileClip(audio_path)
    
    # 비디오에 오디오 추가
    video_with_audio = video_clip.set_audio(audio_clip)
    
    # 새로운 비디오 파일 저장
    video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

# 파일 경로 설정
video_file = "silent_video.mp4"  # 소리 없는 MP4 파일 경로
audio_file = "audio.wav"         # WAV 파일 경로
output_file = "output_video.mp4" # 출력 파일 경로

# 비디오와 오디오 합치기
merge_audio_video(video_file, audio_file, output_file)
