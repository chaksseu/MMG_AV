import os
import pandas as pd
import re

# mp4 파일들이 위치한 폴더 경로 설정
folder_path = '/home/jupyter/preprocessed_WebVid_10M_train_videos_0130'
output_csv = '/home/jupyter/preprocessed_WebVid_10M_videos_0204.csv'

# 폴더 내 모든 mp4 파일 목록 가져오기
file_list = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

# id와 caption을 저장할 리스트 초기화
ids = []
captions = []
num_test = 5000

for filename in file_list:
    ids.append(filename)
    
    # 정규 표현식을 사용하여 앞 숫자 부분을 제거
    caption = re.sub(r'^\d+_', '', filename).rsplit('.mp4', 1)[0]
    
    # 언더스코어를 공백으로 변환
    caption = caption.replace('_', ' ')
    
    captions.append(caption)

# 데이터프레임 생성
df = pd.DataFrame({
    'id': ids,
    'caption': captions
})

# 데이터프레임을 랜덤하게 섞기
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# split 열 추가: 상위 num_test 개는 'test', 나머지는 'train'
df['split'] = ['test' if i < num_test else 'train' for i in range(len(df))]

# CSV 파일로 저장
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"CSV 파일이 '{output_csv}'로 저장되었습니다.")
