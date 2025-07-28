import os
import zipfile
from huggingface_hub import HfApi

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, arcname=rel_path)

def upload_zipped_folder_to_hf(folder_path, repo_id):
    api = HfApi()

    # 압축할 zip 파일 경로
    zip_path = folder_path.rstrip('/').rstrip('\\') + ".zip"

    # 폴더 압축
    if zip_path and os.path.exists(zip_path):
        print("압축 파일이 이미 존재합니다. 건너뜁니다.")
        pass
    else:
        zip_folder(folder_path, zip_path)
        print(f"📦 폴더 압축 완료: {zip_path}")

    # Hugging Face Hub에 업로드
    api.upload_file(
        path_or_fileobj=zip_path,
        path_in_repo=os.path.basename(zip_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"✅ {zip_path} 업로드 완료")

# 예시 사용
repo_id = "Chaksseu/mmg_inference_foler_0326"
folder_path = "/home/work/kby_hgh/MMG_Inferencce_folder/tensorboard"

upload_zipped_folder_to_hf(folder_path, repo_id)
