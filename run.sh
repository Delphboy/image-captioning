#!/bin/bash

source .venv/bin/activate

python3 train.py --exp_name "DEBUG" \
		--captions_file "/home/henry/Datasets/coco/dataset_coco_20.json" \
		--butd_root "/home/henry/Datasets/coco/butd_att/" \
		--sgae_root "/home/henry/Datasets/coco/coco_img_sg/" \
		--vsua_root "/home/henry/Datasets/coco/geometry-iou-iou0.2-dist0.5-undirected/" \
		--input_mode "butd" \
		--feature_limit 50 \
		--token_dim 2048 \
		--enc_model_type "none" \
		--enc_num_layers 1 \
		--dec_lang_model "lstm" \
		--dec_num_layers 1 \
		--warm_up 10000 \
		--batch_size 64 \
		--epochs 30 \
		--force_rl_after -1 \
		--learning_rate 5e-4 \
		--workers 4 \
		--seed -1 \
		--patience -1 \
		--checkpoint_location "checkpoints/" \
		--beam_width 3 \
		--dropout 0.1
