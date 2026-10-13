# Completed ViT-S training-seed sweep

Run shell entry points from the login node; all Python and dataset I/O run in Slurm.

```bash
# Inventory, meaningful aggregation tests, and submission commands without GPU submission:
bash eval_jobs/submit_vits_sweep.sh bootstrap --dry-run
# Start the workflow (once):
bash eval_jobs/submit_vits_sweep.sh
```

The bootstrap stages the existing ImageNet binary caches and ADE20K archive, including
the repository's required train/val split lists, in a CPU job. It validates all listed image/mask pairs,
verifies all 14 final teacher checkpoints and 224/512 features in one GPU job, then launches
both `vits_claude_seed0` evaluations. Only successful, compatible pilots release the
remaining 26 evaluations. Each task array allows at most two concurrent jobs. Slurm
dependencies schedule the continuation and final aggregation without polling or sleeps.

Linear probing uses evaluation seed 0, frozen last-block CLS and mean patch features,
16 learning rates, batch 128, 100100 iterations, 256/224 transforms, and 12/8 workers.
Segmentation uses evaluation seed 100, a frozen backbone and linear head, batch 16,
FP32, 40000 iterations, 6 workers, and validation at steps 20000 and 40000. The standard
segmentation trainer is wrapped to save the decoder and all metrics from the validation
with best mIoU. An earlier best validation is retained even if another metric improves later.

GPU requests: account `aip-papyan`, one `l40s`, 16 CPUs; linear 128G/3h and segmentation
24G/4h45m. No partition is specified. All launches use standalone single-node torchrun.
The scripts explicitly load StdEnv/2023, gcc/12.3, python/3.13.2, arrow/19.0.1 and the
existing dinov3 environment. ADE20K unpacks only into `$SLURM_TMPDIR`.

Outputs and logs: `$SCRATCH/dinov3_results/vits_claude_sweep_eval`. The inventory includes
all 15 training runs, completion evidence, restart/final checkpoint paths, task status,
and Slurm array job IDs. `vits_6L_claude_seed2` remains excluded and untouched.

```bash
# Retry one failed task independently; completed compatible results are skipped:
bash eval_jobs/submit_vits_sweep.sh submit --phase retry \
  --runs vits_claude_seed0 --tasks segmentation
# After repairing a pilot failure, re-run the gated continuation:
bash eval_jobs/submit_vits_sweep.sh advance
# Refresh reports and inventory at any time:
bash eval_jobs/submit_vits_sweep.sh aggregate
# Re-run checks or checkpoint verification when needed:
bash eval_jobs/submit_vits_sweep.sh test
```

For GPU re-verification, use the resource flags emitted by the dry run for `sweep_gpu.sh verify`.
Do not change source during running evaluations: the provenance check intentionally rejects
results from different evaluation code or configurations. Exact scheduling-only changes
may be accepted through an audited source-hash migration in `scheduling_compatibility.json`;
the original verification and attempt fingerprints remain intact. Each retry writes a fresh attempt
directory. The retry utility checks submitted jobs to avoid concurrent duplicate submissions.
Multiple compatible completed attempts for the same run/task are reported as duplicates and
excluded until reviewed. Failed, malformed, unfinished, missing, and incompatible attempts
are explicitly listed in `excluded.json`.

Validated compact reports are copied to `eval_results/vits_claude_sweep/`: `per_run.csv`,
`summary.csv`, `summary.json`, `inventory.csv`, `inventory.json`, and `excluded.json`.
Means and sample standard deviations (`ddof=1`) remain in percentage units. Coverage is
complete only at n=3; 6L has n=2 when its two finished runs succeed. Empty means and
standard deviations with fewer than two observations are null. Reports generated while
jobs run are partial and clearly show missing evaluations; final aggregation runs after
both arrays terminate, including failed jobs.
