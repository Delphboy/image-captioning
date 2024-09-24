#!/bin/bash
#$ -l h_rt=24:00:00
#$ -l h_vmem=7.5G
#$ -pe smp 12
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -wd /data/home/eey362/code/image-captioning/
#$ -j y
#$ -m ea
#$ -o logs/
#$ -l cluster=andrena

module purge

module load python/3.10.7
module load cuda/11.8.0
module load gcc/6.3.0
module load java/1.8.0_382-openjdk

source .venv/bin/activate

python3 train.py --exp_name "butd-test" \
		--captions_file "/data/EECS-YuanLab/COCO/dataset_coco.json" \
		--butd_root "/data/EECS-YuanLab/COCO/butd_att/" \
		--sgae_root "/data/EECS-YuanLab/COCO/coco_img_sg/" \
		--vsua_root "/data/EECS-YuanLab/COCO/geometry-iou-iou0.2-dist0.5-undirected/" \
		--input_mode "butd" \
		--feature_limit 50 \
		--token_dim 1000 \
		--enc_model_type "none" \
		--enc_num_layers 0 \
		--dec_lang_model "dual_lstm" \
		--dec_num_layers 1 \
		--batch_size 64 \
		--epochs 34 \
		--force_rl_after -1 \
		--learning_rate 5e-4 \
		--workers 4 \
		--seed -1 \
		--patience -1 \
		--checkpoint_location "/data/scratch/eey362/image-captioning-checkpoints/" \
		--beam_width 5 \
		--dropout 0.1 \
