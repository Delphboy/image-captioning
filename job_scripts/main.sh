#!/bin/bash
#$ -l h_rt=24:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -wd /data/home/eey362/image-captioning
#$ -j y
#$ -N main
#$ -m bea

cd /data/home/eey362/image-captioning

module load python/3.8.5
module load cudnn/8.1.1-cuda11
module load cuda/11.0.3

python -m venv .venv
source .venv/bin/activate

# python -m pip install --upgrade pip

pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install tqdm
pip install pycocotools
pip install pandas
pip install -r /data/home/eey362/image-captioning/requirements.txt

echo ""
echo ""
echo ""
echo ""
echo ""
pip freeze

# Run!
python main.py
