#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=80G
#SBATCH --time=00:05:00
#SBATCH --job-name=block_pca
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

set -euo pipefail

REPO_DIR=/home/yucz/projects/aip-papyan/yucz/dinov3
cd "${REPO_DIR}"

module load gcc arrow/21.0.0

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

MODEL_NAME=${1:-}

if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: sbatch measurements/run_block_pca.sh <model_name>" >&2
    exit 2
fi

PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" python -m measurements.plot_block_pca \
    --checkpoint "${MODEL_NAME}/eval/training_124999/teacher_checkpoint.pth" \
    --output-dir "measurements/pca_plots"
