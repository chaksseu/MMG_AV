conda update -n base -c defaults conda -y
conda init
conda config --add envs_dirs /workspace/conda/envs
conda config --add pkgs_dirs /workspace/conda/pkgs
conda config --set croot /workspace/conda/conda-bld
apt-get update
apt-get upgrade -y
apt-get install ffmpeg wget git zip unzip gcc libopenmpi-dev libmpich-dev curl tar libjpeg-dev libpng-dev libgl1-mesa-glx libglib2.0-0 libsndfile1 ninja-build -y
apt-get update
apt-get upgrade -y
conda activate MMG
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#pip install -U pip
#pip install -r requirements.txt
export HF_HOME=/workspace/huggingface_cache