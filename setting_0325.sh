sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install ffmpeg wget git zip unzip gcc libopenmpi-dev libmpich-dev curl tar libjpeg-dev libpng-dev libgl1-mesa-glx libglib2.0-0 libsndfile1 ninja-build -y
sudo apt-get update
sudo apt-get upgrade -y
bash Miniconda3-latest-Linux-x86_64.sh

conda init
pip install -U pip
pip install nvitop
conda create -n mmg python=3.10 -y

conda activate mmg
cd MMG_01

pip install -U pip
pip install nvitop
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
# conda install -c conda-forge mpi4py -y
# conda install cuda-cudart cuda-version=12 -y
conda install -c conda-forge moviepy -y
pip install -r requirements.txt
conda install numpy==1.23.5 scipy==1.11.1 -y
# pip install flash-attn --no-build-isolation
# pip install decord ffmpeg-python imageio opencv-python==4.8.0.74

accelerate config

pip install soxr

nano /home/work/miniconda3/envs/mmg/lib/python3.10/site-packages/pytorchvideo/transforms/augmentations.py
# delete _tensor
nano /home/work/miniconda3/envs/mmg/lib/python3.10/site-packages/laion_clap/clap_module/factory.py
# torch.load -> , weights_only=False

nano /home/work/miniconda3/envs/mmg/lib/python3.10/site-packages/laion_clap/clap_module/model.py
# /home/work/kby_hgh/roberta-base
# /home/work/kby_hgh/bart_local
nano /home/work/miniconda3/envs/mmg/lib/python3.10/site-packages/laion_clap/clap_module/bert.py
huggingface-cli login
