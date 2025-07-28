import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

def load_mnist_folder_as_rgb_tensor(folder, image_size=(299, 299), max_images=None):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),  # (1CH → 3CH)
        transforms.ToTensor(),  # [0,1]
    ])
    images = []
    files = sorted(os.listdir(folder))
    for i, fname in enumerate(files):
        if not fname.endswith('.png'):
            continue
        if max_images and i >= max_images:
            break
        img = Image.open(os.path.join(folder, fname)).convert("L")  # load as grayscale
        img = transform(img)  # shape: [3, 299, 299]
        images.append(img)
    return torch.stack(images)

# 예시 경로 설정
real_dir = "/home/work/kby_hgh/MMG_01/toy_mmg/MNIST/train/0"
fake_dir = "/home/work/kby_hgh/MMG_01/toy_mmg/MNIST/train/0"

# 이미지 텐서 로딩 (각각 [N, 3, 299, 299])
real_imgs = load_mnist_folder_as_rgb_tensor(real_dir).to("cpu")
fake_imgs = load_mnist_folder_as_rgb_tensor(fake_dir).to("cpu")

print(f"Loaded real: {len(real_imgs)}, fake: {len(fake_imgs)}")

# FID 측정
fid = FrechetInceptionDistance(normalize=True).to("cpu")
fid.update(real_imgs, real=True)
fid.update(fake_imgs, real=False)

print(f"FID (MNIST PNG folders): {float(fid.compute()):.4f}")