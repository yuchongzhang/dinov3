#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=linear
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

module load gcc arrow/19.0.1

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

PYTHONPATH=${PWD}:${PYTHONPATH} python -m dinov3.eval.linear \
  model.config_file=/home/yucz/projects/aip-papyan/yucz/dinov3/vits_1L_claude/config.yaml \
  model.pretrained_weights=/home/yucz/projects/aip-papyan/yucz/dinov3/vits_1L_claude/eval/training_124999/teacher_checkpoint.pth \
  output_dir=/home/yucz/projects/aip-papyan/yucz/dinov3/vits_1L_claude/linear_eval_results/ \
  train.dataset=ImageNetParquet:split=TRAIN:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data \
  train.val_dataset=ImageNetParquet:split=VAL:root=/home/yucz/projects/aip-papyan/shared/imagenet-1k/data