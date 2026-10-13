#!/bin/bash
# Source only inside a Slurm allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this workflow through Slurm}"
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.13.2 arrow/19.0.1
source /project/aip-papyan/yucz/venvs/dinov3_env/bin/activate
export SWEEP_ROOT="${SWEEP_ROOT:-$SCRATCH/dinov3_results/vits_claude_sweep_eval}"
export HF_HOME="$SWEEP_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$SWEEP_ROOT/cache"
export PYTHONPYCACHEPREFIX="$SLURM_TMPDIR/pycache"
export PYTHONPATH="${SWEEP_REPO:?}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
cd "$SWEEP_REPO"
