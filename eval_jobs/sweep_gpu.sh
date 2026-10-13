#!/bin/bash
# Resources and task array are specified by the submission utility.
set -euo pipefail
source "$SWEEP_REPO/eval_jobs/sweep_env.sh"
if [[ "$1" == verify ]]; then
  exec torchrun --standalone --nnodes=1 --nproc-per-node=1 eval_jobs/vits_sweep.py verify
fi
exec python eval_jobs/vits_sweep.py run --task "$1" --manifest "$2"
