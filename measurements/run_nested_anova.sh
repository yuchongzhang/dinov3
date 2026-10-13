#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=aip-papyan
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=80G
#SBATCH --time=00:02:00
#SBATCH --job-name=nested_anova
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

set -euo pipefail

REPO_DIR=/home/yucz/projects/aip-papyan/yucz/dinov3
cd "${REPO_DIR}"

module load gcc arrow/21.0.0

source /home/yucz/projects/aip-papyan/yucz/venvs/dinov3_env/bin/activate

MODEL_NAME=${1:-vits_claude}
LOCAL_THRESHOLD=${2:-3}
K_GLOBAL=${3:-20000}

if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: sbatch measurements/run_nested_anova.sh <model_name> [local_threshold] [k_global]" >&2
    exit 2
fi

PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" python -m measurements.compute_nested_anova \
    --checkpoint "${MODEL_NAME}/eval/training_124999/teacher_checkpoint.pth" \
    --output "measurements/nested_anova_results_test/${MODEL_NAME}_nested_anova_threshold=${LOCAL_THRESHOLD}_k_global=${K_GLOBAL}.json" \
    --records-cache "measurements/nested_anova_records_test/${MODEL_NAME}_records.pt" \
    --batch-size 256 \
    --exclude-cls \
    --local-distance-threshold "${LOCAL_THRESHOLD}" \
    --k-global "${K_GLOBAL}" \
    --max-images 5000
