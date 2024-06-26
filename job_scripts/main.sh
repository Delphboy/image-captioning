#!/bin/bash
#$ -l h_rt=24:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -wd /data/home/eey362/image-captioning
#$ -j y
#$ -m ea
#$ -o log_files/
### $ -l cluster=andrena

cd /data/home/eey362/image-captioning

module load python/3.8.5
module load cuda/11.6.2
module load cudnn/8.4.1-cuda11.6
module load java/1.8.0_241-oracle

# python -m venv .venv
source .venv/bin/activate

# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# pip install torch_geometric
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

# pip install -r /data/home/eey362/image-captioning/requirements.txt

# python3 -m pip freeze > requirements.txt

# Run!
python main.py --file $1