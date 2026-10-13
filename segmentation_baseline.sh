#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --job-name=segmentation
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

module load gcc arrow/19.0.1

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

MODEL="vits_claude"

PYTHONPATH=${PWD}:${PYTHONPATH} python -m dinov3.eval.segmentation.run \
  model.pretrained_weights=/home/yucz/projects/aip-papyan/yucz/dinov3/${MODEL}/eval/training_124999/teacher_checkpoint.pth \
  model.config_file=/home/yucz/projects/aip-papyan/yucz/dinov3/${MODEL}/config.yaml \
  n_gpus=1 \
  bs=16 \
  config=dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
  datasets.root=/home/yucz/projects/aip-papyan/shared/ADE20K/ADEChallengeData2016 \
  output_dir=/home/yucz/projects/aip-papyan/yucz/dinov3/${MODEL}/segmentation_eval_results/