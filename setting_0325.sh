# sudo apt-get update
# sudo apt-get upgrade -y
# sudo apt-get install ffmpeg wget git zip unzip gcc libopenmpi-dev libmpich-dev curl tar libjpeg-dev libpng-dev libgl1-mesa-glx libglib2.0-0 libsndfile1 ninja-build -y
# sudo apt-get update
# sudo apt-get upgrade -y
# bash Miniconda3-latest-Linux-x86_64.sh

conda init
conda create -n mmg python=3.10 -y
conda activate mmg

pip install -U pip
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
# conda install -c conda-forge mpi4py -y
# conda install cuda-cudart cuda-version=12 -y
conda install -c conda-forge moviepy -y
pip install -r requirements.txt

accelerate config

# nano /home/work/miniconda3/envs/mmg/lib/python3.10/site-packages/pytorchvideo/transforms/augmentations.py
# delete _tensor?