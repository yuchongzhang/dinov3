#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=48G
#SBATCH --job-name=segm
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
DEFAULT_SEED=100
SEED=${SEED:-${DEFAULT_SEED}}
SUFFIX=""
[ "${SEED}" != "${DEFAULT_SEED}" ] && SUFFIX="_seed${SEED}"

echo "MODEL=${MODEL}  CKPT=${CKPT_ITER}  SEED=${SEED}"

# eval.eval_interval bumped 5000 -> 20000: on 1 GPU a full ADE20K val pass (2000
# images, slide inference, fp32) takes ~34 min, so the config's 8 validations would
# cost 4.5 h on top of 4.2 h of training. 20000 keeps a mid-training check plus the
# final one (the LR anneals to 0, so the last step is the best one anyway).
PYTHONPATH=${PWD}:${PYTHONPATH:-} python -m dinov3.eval.segmentation.run \
  model.pretrained_weights=${RESULTS_ROOT}/${MODEL}/eval/${CKPT_ITER}/teacher_checkpoint.pth \
  model.config_file=${RESULTS_ROOT}/${MODEL}/config.yaml \
  n_gpus=1 \
  bs=16 \
  config=dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
  datasets.root=/home/yucz/projects/aip-papyan/shared/ADE20K/ADEChallengeData2016 \
  eval.eval_interval=20000 \
  seed=${SEED} \
  output_dir=${RESULTS_ROOT}/${MODEL}/segmentation_eval_results_${CKPT_ITER}${SUFFIX}/
