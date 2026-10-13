#!/bin/bash
# Launch/resume the five vits_*_claude configs x 3 seeds = 15 runs.
#
#   vits_claude     vit_small        (baseline, no Laplacian heads)
#   vits_1L_claude  vit_small_mixed  num_vanilla_heads=5 -> 1 Laplacian head
#   vits_3L_claude  vit_small_mixed  num_vanilla_heads=3 -> 3 Laplacian heads
#   vits_5L_claude  vit_small_mixed  num_vanilla_heads=1 -> 5 Laplacian heads
#   vits_6L_claude  vit_small_mixed  num_vanilla_heads=0 -> 6 Laplacian heads
#
# Scheduling strategy (Vulcan is heavily contested: ~520 GPU jobs ahead of ours):
#
#   * All gpubase_* partitions are the SAME hardware, nested by walltime tier,
#     with identical PriorityTier=20 -- a shorter tier costs no priority:
#         b1  3h -> 231 nodes      b2 12h -> 217 nodes
#         b3 24h -> 148 nodes      b4  3d -> 148        b5 7d -> 28
#     Slurm reserves the full REQUESTED walltime, so a shorter request reaches a
#     larger node pool. But shorter is NOT free here: train.py checkpoints only
#     on the periodic boundary (train.py:600, checkpointing.period=4290 iters
#     ~= 65 min) and has no save-on-SIGUSR2, so every timeout discards up to a
#     full interval. A 12h segment is the worst case for the resuming runs: it
#     ends at iter ~141567, just 3 short of the 141570 checkpoint, throwing away
#     ~1.09h and forcing a second segment for the last ~1.5h of work.
#
#     So every job requests 24h. Resuming runs (12.4h left) then finish in ONE
#     segment with zero loss; never-run runs (36.4h) take two segments instead
#     of four. The cost is the node pool: b3 has 148 nodes vs b2's 217, and the
#     69 nodes that live only in b1/b2 become unreachable.
#
#   * We do not set --partition: the site's Lua router places the job from
#     --timeout. 1440 min -> b3.
#
#   * cpus-per-task is 12, not 16. 4 tasks x 16 = 64 = a whole node's CPUs, and
#     only ~7 nodes cluster-wide have 64 idle CPUs; at 48 total, ~104 do.
#     num_workers=10 per task and measured data time is ~0.0002 s/iter, so
#     12 CPUs/task leaves the dataloader ample headroom.
#
# Every run resumes from its own ckpt/ (train.py:438), so re-running this script
# after cancelling is safe and never restarts finished work. wandb ids derive
# from output_dir, so requeued segments stitch into one run.
#
# RESUME_ITER / REMAIN_h below are printed for visibility only -- they no longer
# select a tier, since every job now requests the same 24h.
#
# Usage:  ./pretrain_imagenet_vits_sweep.sh            # submit all 15
#         DRY_RUN=1 ./pretrain_imagenet_vits_sweep.sh  # print the plan only

set -euo pipefail

CONFIG_DIR=dinov3/configs/train
# Checkpoints live in the shared project dir (~6 GB per run x 15 runs, counts
# against the aip-papyan project quota, not /home).
RUN_ROOT=/home/yucz/projects/aip-papyan/shared/vits_claude_sweep

MODELS=(vits_claude vits_1L_claude vits_3L_claude vits_5L_claude vits_6L_claude)
SEEDS=(0 1 2)

MAX_ITER=143000          # optim.epochs 100 x OFFICIAL_EPOCH_LENGTH 1430
SEC_PER_ITER=0.9155      # measured: 94379 iters in one 24h segment
TIMEOUT=1440             # 24h for every job -> gpubase_bygpu_b3

printf "%-26s %11s %9s %8s  %s\n" RUN RESUME_ITER REMAIN_h TIMEOUT SEGMENTS
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="${model}_seed${seed}"
    config="${CONFIG_DIR}/${model}.yaml"
    out_dir="${RUN_ROOT}/${run_name}"

    if [ ! -f "${config}" ]; then
      echo "ERROR: missing config ${config}" >&2
      exit 1
    fi

    # Where this run will resume from, so we can right-size its walltime.
    last_ckpt=$(ls "${out_dir}/ckpt" 2>/dev/null | sort -n | tail -1 || true)
    start_iter=${last_ckpt:-0}
    remain_h=$(awk -v i="${start_iter}" -v m="${MAX_ITER}" -v s="${SEC_PER_ITER}" \
                   'BEGIN{printf "%.1f", (m-i)*s/3600}')

    # How many 24h segments this run still needs, for the plan printout.
    segments=$(awk -v r="${remain_h}" 'BEGIN{n=int(r/24); if(r>n*24) n++; print n}')

    printf "%-26s %11s %9s %8s  %s\n" "${run_name}" "${start_iter}" "${remain_h}" \
           "${TIMEOUT}" "b3 24h (${segments} segment(s))"

    cmd=(python -m dinov3.run.submit dinov3/train/train.py
      --nodes 1
      --ngpus 4
      --cpus-per-task 12
      --mem-gb 256
      --timeout "${TIMEOUT}"
      --slurm-account "aip-papyan"
      --slurm-setup "module load gcc arrow/19.0.1"
      --config-file "${config}"
      --output-dir "${out_dir}"
      --seed "${seed}"
      train.seed="${seed}"
      wandb.name="${run_name}")

    if [ -n "${DRY_RUN:-}" ]; then
      echo "    ${cmd[*]}"
    else
      PYTHONPATH=${PWD} "${cmd[@]}"
    fi
  done
done
