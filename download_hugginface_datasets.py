from huggingface_hub import snapshot_download
import os

# 저장할 디렉터리 설정
download_dir = "/home/work/kby_hgh/norispace-project/0522_datasets"
os.makedirs(download_dir, exist_ok=True)

# 데이터셋 다운로드
dataset_name = "Chaksseu/0522_datasets"
snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=download_dir) #, force_download=True)

# 데이터셋 파일 저장 경로 출력
print(f"Dataset downloaded to: {download_dir}")
