#!/bin/bash

set -euo pipefail

MODEL_NAMES=("vits_6L_claude" "vits_5L_claude" "vits_3L_claude" "vits_1L_claude" "vits_claude")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    sbatch measurements/run_anova.sh "${MODEL_NAME}"
done
