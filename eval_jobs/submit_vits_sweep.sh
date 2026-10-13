#!/bin/bash
# Lightweight entry point: Python, staging, and all evaluations run on compute nodes.
set -euo pipefail
SWEEP_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
export SWEEP_REPO
export SWEEP_ROOT="${SWEEP_ROOT:-$SCRATCH/dinov3_results/vits_claude_sweep_eval}"
mkdir -p "$SWEEP_ROOT/logs"
args=(bootstrap)
if (( $# )); then args=("$@"); fi
exec sbatch --parsable --export=ALL --chdir="$SWEEP_REPO" \
  --output="$SWEEP_ROOT/logs/%x_%j.out" --error="$SWEEP_ROOT/logs/%x_%j.err" \
  "$SWEEP_REPO/eval_jobs/sweep_cpu.sh" "${args[@]}"
