#!/bin/bash
#SBATCH --account=aip-papyan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=vits_sweep_cpu
set -euo pipefail
source "$SWEEP_REPO/eval_jobs/sweep_env.sh"
exec python eval_jobs/vits_sweep.py "$@"
