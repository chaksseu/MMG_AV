import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd



class VideoAudioTextDataset(Dataset):
    def __init__(self, csv_path, video_dir, audio_dir, transform=None):
        """
        Args:
            csv_path (str): dataset_info.csv 파일 경로
            video_dir (str): 비디오 텐서 폴더 경로
            audio_dir (str): 오디오 텐서 폴더 경로
            transform (callable, optional): video나 audio 텐서에 적용할 transform 함수. 
        """
        self.csv_path = csv_path
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.transform = transform

        # CSV 로드
        self.data = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # csv에서 한 행을 가져오기
        row = self.data.iloc[idx]

        # video 파일 경로
        video_name = row['Video']
        video_path = os.path.join(self.video_dir, video_name)

        # audio 파일 경로
        audio_name = row['Audio']
        audio_path = os.path.join(self.audio_dir, audio_name)

        # text
        text = row['Text']

        # pt파일 로드 (torch.load로 텐서 로드)
        video_tensor = torch.load(video_path)  # [C,T,H,W] 형태 가정 (예시)
        audio_tensor = torch.load(audio_path)  # [C,T] 형태 가정 (예시)
        
        # 필요하다면 transform 적용
        if self.transform:
            video_tensor = self.transform(video_tensor)
            audio_tensor = self.transform(audio_tensor)

        return video_tensor, audio_tensor, text

# 사용 예시:
if __name__ == "__main__":
    csv_file = "../_processed_data/dataset_info.csv"
    video_folder = "../_processed_data/video"
    audio_folder = "../_processed_data/audio"

    dataset = VideoAudioTextDataset(csv_path=csv_file, 
                                    video_dir=video_folder, 
                                    audio_dir=audio_folder,
                                    transform=None)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 데이터 로딩 예시
    for batch_idx, (videos, audios, texts) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("Videos shape:", videos.shape) # Videos shape: torch.Size([B, 64, 3, 256, 256])
        print("Audios shape:", audios.shape) # Audios shape: torch.Size([B, 3, 256, 400])
        print("Texts:", texts)
        if batch_idx > 1:
            break
