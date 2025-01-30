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
#conda create -n mmg python=3.10 -y
conda activate mmg
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
#pip install -U pip
#pip install -r requirements.txt
#chown -R 1001:1001 /workspace
#conda install -c conda-forge mpi4py -y
#conda install cuda-cudart cuda-version=12 -y
###conda install -c conda-forge moviepy -y
chown -R 1001:1001 /workspace
export HF_HOME=/workspace/huggingface_cache
