#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=48G
#SBATCH --job-name=segm_vits
#SBATCH --output=logs_eval/%j_%x.out
#SBATCH --error=logs_eval/%j_%x.err

# `sbatch --export=ALL,MODEL=<name>,SEED=<n> --time=...`
# Segmentation hyperparameters are those of the original segmentation.sh /
# config-ade20k-linear-training.yaml: total batch 16 (bs=16 on 1 GPU), 40000 iters,
# AdamW lr 1e-3, WarmupOneCycleLR, fp32, linear head on the last block, 512 crops.
# Only eval_interval is changed (5000 -> 20000) to cut validation cost; see below.

set -euo pipefail

module load gcc arrow/19.0.1

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

cd /home/yucz/projects/aip-papyan/yucz/dinov3

RESULTS_ROOT=/home/yucz/projects/aip-papyan/yucz/dinov3
CKPT_ITER=training_124999
SEED=${SEED:-100}

echo "MODEL=${MODEL}  CKPT=${CKPT_ITER}  SEED=${SEED}"

# On 1 GPU a full ADE20K val pass is expensive (the config is written for 8 GPUs), so
# validating every 5000 steps would dominate the job. 20000 keeps a mid-training check
# plus the final one; the LR anneals to 0, so the last step is the best one anyway.
PYTHONPATH=${PWD}:${PYTHONPATH:-} python -m dinov3.eval.segmentation.run \
  model.pretrained_weights=${RESULTS_ROOT}/${MODEL}/eval/${CKPT_ITER}/teacher_checkpoint.pth \
  model.config_file=${RESULTS_ROOT}/${MODEL}/config.yaml \
  n_gpus=1 \
  bs=16 \
  config=dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
  datasets.root=/home/yucz/projects/aip-papyan/shared/ADE20K/ADEChallengeData2016 \
  eval.eval_interval=20000 \
  seed=${SEED} \
  output_dir=${RESULTS_ROOT}/${MODEL}/segmentation_eval_results_${CKPT_ITER}_seed${SEED}/
