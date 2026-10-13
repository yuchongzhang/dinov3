#!/bin/bash

 PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 1 \
  --ngpus 4 \
  --cpus-per-task 16 \
  --mem-gb 256 \
  --timeout 1440 \
  --slurm-partition "gpubase_bygpu_b3" \
  --slurm-account "aip-papyan" \
  --slurm-setup "module load gcc arrow/19.0.1" \
  --config-file dinov3/configs/train/vitb_gpt/vitb_6L_gpt.yaml \
  --output-dir /home/yucz/scratch/dinov3_results/vitb_6L_gpt \
  train.dataset_path=ImageNetParquet:split=TRAIN:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data
