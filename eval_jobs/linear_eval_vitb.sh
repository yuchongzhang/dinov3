#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=96G
#SBATCH --job-name=linear
#SBATCH --output=logs_eval/%j_%x.out
#SBATCH --error=logs_eval/%j_%x.err

# MODEL is passed in via `sbatch --export=ALL,MODEL=<name>`
# Time limit is passed in via `sbatch --time=...`

set -euo pipefail

module load gcc arrow/19.0.1

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

cd /home/yucz/projects/aip-papyan/yucz/dinov3

RESULTS_ROOT=/home/yucz/scratch/dinov3_results
CKPT_ITER=training_499999

# `sbatch --export=ALL,MODEL=<name>,SEED=<n>` to run a seed sweep. Non-default seeds
# get their own output dir so runs of the same model do not overwrite each other.
DEFAULT_SEED=0
SEED=${SEED:-${DEFAULT_SEED}}
SUFFIX=""
[ "${SEED}" != "${DEFAULT_SEED}" ] && SUFFIX="_seed${SEED}"

echo "MODEL=${MODEL}  CKPT=${CKPT_ITER}  SEED=${SEED}"

# This eval is bottlenecked by random reads into the 147 GB ImageNet mmap cache on
# the shared FS (~17 ms latency, so throughput scales with in-flight reads), not by
# the GPU. Measured on one node: 8 workers -> 0.48 s/iter (data 0.42), 16 workers ->
# 0.073 s/iter (data 0.003). A busier node still starved at 16 workers (0.24 s/iter),
# hence 48 workers over 32 cpus: ~2800 img/s of I/O concurrency, and enough cores to
# JPEG-decode it. GPU-bound floor is ~0.073 s/iter.
PYTHONPATH=${PWD}:${PYTHONPATH:-} python -m dinov3.eval.linear \
  model.config_file=${RESULTS_ROOT}/${MODEL}/config.yaml \
  model.pretrained_weights=${RESULTS_ROOT}/${MODEL}/eval/${CKPT_ITER}/teacher_checkpoint.pth \
  output_dir=${RESULTS_ROOT}/${MODEL}/linear_eval_results_${CKPT_ITER}${SUFFIX}/ \
  seed=${SEED} \
  train.num_workers=48 \
  eval.num_workers=32 \
  train.dataset=ImageNetParquet:split=TRAIN:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data \
  train.val_dataset=ImageNetParquet:split=VAL:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data
