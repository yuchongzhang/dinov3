#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=80G
#SBATCH --time=00:26:00
#SBATCH --job-name=anova
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

set -euo pipefail

REPO_DIR=/home/yucz/projects/aip-papyan/yucz/dinov3
cd "${REPO_DIR}"

module load gcc arrow/21.0.0

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

MODEL_NAME=${1:-}

if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: sbatch measurements/run_anova.sh <model_name>" >&2
    exit 2
fi

PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" python -m measurements.compute_anova \
    --checkpoint "${MODEL_NAME}/eval/training_124999/teacher_checkpoint.pth" \
    --output "measurements/anova_results_no_cls/${MODEL_NAME}_anova.json" \
    --batch-size 512 \
    --exclude-cls
