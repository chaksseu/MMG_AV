import os
import pandas as pd

# mp4 파일들이 위치한 폴더 경로를 설정하세요
folder_path = 'preprocessed_WebVid_10M_train_videos_0125'  # 예: 'C:/videos'
output_csv = 'preprocessed_WebVid_10M_0125_2097.csv'  # 원하는 출력 파일 이름으로 변경 가능

# 폴더 내 모든 mp4 파일 목록 가져오기
file_list = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

# id와 caption을 저장할 리스트 초기화
ids = []
captions = []
num_test = 100


for filename in file_list:
    ids.append(filename)
    # 'name_' 뒤와 '.mp4' 앞 사이의 문자열을 caption으로 추출
    try:
        caption = filename.split('_name_')[1].rsplit('.mp4', 1)[0]
    except IndexError:
        caption = ''
    captions.append(caption)

# 데이터프레임 생성
df = pd.DataFrame({
    'id': ids,
    'caption': captions
})

# 데이터프레임을 랜덤하게 섞기 (재현 가능성을 위해 random_state 설정)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# split 열 추가: 상위 5000개는 'test', 나머지는 'train'
df['split'] = ['test' if i < num_test else 'train' for i in range(len(df))]

# CSV 파일로 저장
df.to_csv(output_csv, index=False, encoding='utf-8-sig')  # 한글 깨짐 방지를 위해 encoding 설정
print(f"CSV 파일이 '{output_csv}'로 저장되었습니다.")
