#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --job-name=spectrum
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

set -euo pipefail

cd /home/yucz/projects/aip-papyan/yucz/dinov3

module load gcc arrow/19.0.1

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

python measurements/image_spectrum_dino.py --center-per-image --device cuda --n-images -1 --plot-only
