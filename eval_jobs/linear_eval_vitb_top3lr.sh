#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=96G
#SBATCH --job-name=linear_top3
#SBATCH --output=logs_eval/%j_%x.out
#SBATCH --error=logs_eval/%j_%x.err

# Seed replication restricted to the three best learning rates from the seed-0 sweep.
# Those were nominal 2e-3 / 5e-3 / 1e-2 (effective 0.001 / 0.0025 / 0.005 at batch 128 on
# 1 GPU) and were the top three for all five vitb models, with 5e-3 best in every case.
# Narrowing the grid also narrows the selection-on-val bias from 15 candidates to 3.
# `sbatch --export=ALL,MODEL=<name>,SEED=<n> --time=...`

set -euo pipefail

module load gcc arrow/19.0.1

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

cd /home/yucz/projects/aip-papyan/yucz/dinov3

RESULTS_ROOT=/home/yucz/scratch/dinov3_results
CKPT_ITER=training_499999
SEED=${SEED:-0}

echo "MODEL=${MODEL}  CKPT=${CKPT_ITER}  SEED=${SEED}  LRS=top3"

PYTHONPATH=${PWD}:${PYTHONPATH:-} python -m dinov3.eval.linear \
  model.config_file=${RESULTS_ROOT}/${MODEL}/config.yaml \
  model.pretrained_weights=${RESULTS_ROOT}/${MODEL}/eval/${CKPT_ITER}/teacher_checkpoint.pth \
  output_dir=${RESULTS_ROOT}/${MODEL}/linear_eval_results_${CKPT_ITER}_top3lr_seed${SEED}/ \
  seed=${SEED} \
  train.learning_rates=[0.002,0.005,0.01] \
  train.num_workers=48 \
  eval.num_workers=32 \
  train.dataset=ImageNetParquet:split=TRAIN:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data \
  train.val_dataset=ImageNetParquet:split=VAL:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data
