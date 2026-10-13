#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=96G
#SBATCH --job-name=linear_cls
#SBATCH --output=logs_eval/%j_%x.out
#SBATCH --error=logs_eval/%j_%x.err

# CLS-token-only linear probe: `train.use_avgpool_list=[false]` drops the mean-of-patch-
# tokens half of the feature, so the head sees 768 dims instead of 1536. Everything else
# matches the main vitb runs so the two are directly comparable.
# `sbatch --export=ALL,MODEL=<name>,SEED=<n> --time=...`

set -euo pipefail

module load gcc arrow/19.0.1

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

cd /home/yucz/projects/aip-papyan/yucz/dinov3

RESULTS_ROOT=/home/yucz/scratch/dinov3_results
CKPT_ITER=training_499999
SEED=${SEED:-0}

echo "MODEL=${MODEL}  CKPT=${CKPT_ITER}  SEED=${SEED}  POOLING=cls_only"

PYTHONPATH=${PWD}:${PYTHONPATH:-} python -m dinov3.eval.linear \
  model.config_file=${RESULTS_ROOT}/${MODEL}/config.yaml \
  model.pretrained_weights=${RESULTS_ROOT}/${MODEL}/eval/${CKPT_ITER}/teacher_checkpoint.pth \
  output_dir=${RESULTS_ROOT}/${MODEL}/linear_eval_results_${CKPT_ITER}_clsonly_seed${SEED}/ \
  seed=${SEED} \
  train.use_avgpool_list=[false] \
  train.num_workers=48 \
  eval.num_workers=32 \
  train.dataset=ImageNetParquet:split=TRAIN:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data \
  train.val_dataset=ImageNetParquet:split=VAL:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data
