#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=96G
#SBATCH --job-name=linear_vits
#SBATCH --output=logs_eval/%j_%x.out
#SBATCH --error=logs_eval/%j_%x.err

# `sbatch --export=ALL,MODEL=<name>,SEED=<n> --time=...`
# Eval hyperparameters are the defaults used by the original linear_eval.sh runs
# (batch 128, 10 x 10010 iters, 16-LR sweep, last block, avgpool, 256/224 transforms) --
# confirmed against <model>/linear_eval_results/eval_config.yaml. Only num_workers is
# raised, which affects throughput and not results.

set -euo pipefail

module load gcc arrow/19.0.1

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

cd /home/yucz/projects/aip-papyan/yucz/dinov3

RESULTS_ROOT=/home/yucz/projects/aip-papyan/yucz/dinov3
CKPT_ITER=training_124999
SEED=${SEED:-0}

echo "MODEL=${MODEL}  CKPT=${CKPT_ITER}  SEED=${SEED}"

PYTHONPATH=${PWD}:${PYTHONPATH:-} python -m dinov3.eval.linear \
  model.config_file=${RESULTS_ROOT}/${MODEL}/config.yaml \
  model.pretrained_weights=${RESULTS_ROOT}/${MODEL}/eval/${CKPT_ITER}/teacher_checkpoint.pth \
  output_dir=${RESULTS_ROOT}/${MODEL}/linear_eval_results_${CKPT_ITER}_seed${SEED}/ \
  seed=${SEED} \
  train.num_workers=48 \
  eval.num_workers=32 \
  train.dataset=ImageNetParquet:split=TRAIN:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data \
  train.val_dataset=ImageNetParquet:split=VAL:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data
