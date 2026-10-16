# ViT-S training-seed sweep results

30 of 30 evaluations are complete and valid across 15 finished training runs.
Results refreshed at 2026-10-14T21:16:04Z.
All 40 metric summaries were independently checked against raw results using NumPy mean and sample standard deviation (`ddof=1`), with absolute tolerance 1e-12.

Values are percentages, reported as mean ± sample standard deviation across training seeds. They are descriptive statistics, not confidence intervals.

| Model | Training seeds | n | ImageNet linear accuracy | ADE20K mIoU |
|---|---|---:|---:|---:|
| `vits_claude` | 0, 1, 2 | 3 | 72.08 ± 0.18 | 30.27 ± 0.70 |
| `vits_1L_claude` | 0, 1, 2 | 3 | 73.05 ± 0.27 | 31.05 ± 0.57 |
| `vits_3L_claude` | 0, 1, 2 | 3 | 73.34 ± 0.20 | 32.11 ± 0.54 |
| `vits_5L_claude` | 0, 1, 2 | 3 | 73.12 ± 0.13 | 31.83 ± 0.82 |
| `vits_6L_claude` | 0, 1, 2 | 3 | 72.94 ± 0.09 | 30.04 ± 1.08 |

These small samples do not establish statistical significance between models.

## Per-seed results

| Model | Training seed | ImageNet accuracy (%) | ADE20K mIoU (%) |
|---|---:|---:|---:|
| `vits_claude` | 0 | 72.284 | 30.82 |
| `vits_claude` | 1 | 71.930 | 29.48 |
| `vits_claude` | 2 | 72.030 | 30.50 |
| `vits_1L_claude` | 0 | 73.058 | 30.66 |
| `vits_1L_claude` | 1 | 73.306 | 31.71 |
| `vits_1L_claude` | 2 | 72.776 | 30.78 |
| `vits_3L_claude` | 0 | 73.562 | 32.70 |
| `vits_3L_claude` | 1 | 73.178 | 31.65 |
| `vits_3L_claude` | 2 | 73.274 | 31.99 |
| `vits_5L_claude` | 0 | 73.262 | 31.73 |
| `vits_5L_claude` | 1 | 73.102 | 32.69 |
| `vits_5L_claude` | 2 | 73.010 | 31.06 |
| `vits_6L_claude` | 0 | 72.874 | 30.72 |
| `vits_6L_claude` | 1 | 72.896 | 28.80 |
| `vits_6L_claude` | 2 | 73.044 | 30.60 |

## Additional segmentation metrics

All metrics for a run come from the validation selected by highest mIoU. Values remain percentages.

| Model | aAcc | Mean accuracy / recall | Dice | F-score | Precision |
|---|---:|---:|---:|---:|---:|
| `vits_claude` | 73.57 ± 0.44 | 39.18 ± 0.95 | 42.73 ± 0.80 | 43.41 ± 0.65 | 52.47 ± 0.73 |
| `vits_1L_claude` | 73.89 ± 0.34 | 40.11 ± 0.70 | 43.80 ± 0.67 | 44.19 ± 0.59 | 53.30 ± 0.74 |
| `vits_3L_claude` | 74.47 ± 0.36 | 41.34 ± 0.91 | 45.02 ± 0.73 | 45.63 ± 0.45 | 54.06 ± 0.24 |
| `vits_5L_claude` | 74.20 ± 0.45 | 40.92 ± 0.90 | 44.77 ± 0.99 | 45.17 ± 1.16 | 54.51 ± 1.11 |
| `vits_6L_claude` | 73.24 ± 0.58 | 38.87 ± 1.48 | 42.53 ± 1.30 | 43.00 ± 1.25 | 52.27 ± 0.60 |

## Protocol, exclusions, and artifacts

Every backbone uses its saved training configuration and final `eval/training_142999/teacher_checkpoint.pth`. Linear probing uses evaluation seed 0, frozen last-block CLS and mean patch features, 16 learning rates, batch 128, 100100 iterations, and 256/224 transforms. Segmentation uses evaluation seed 100, a frozen backbone and linear head, batch 16, FP32, 40000 iterations, and best mIoU from validations at 20000 and 40000. Time limits are 03:00:00 for linear probing and 04:45:00 for segmentation.

Failed, cancelled, malformed, unfinished, incompatible, or duplicate attempts are excluded from statistics. See [excluded.json](excluded.json) for attempt history. The source compatibility audit covers inventory/reporting changes only; evaluation configurations and metric selection are unchanged.

- [summary.csv](summary.csv): full precision summaries, contributing seeds, and coverage.
- [summary.json](summary.json): summaries and accepted evaluations with provenance.
- [per_run.csv](per_run.csv): per-run metrics, selected classifiers, and validation steps.
- [inventory.json](inventory.json) and [inventory.csv](inventory.csv): all 15 training runs and evaluation job histories.
- [metrics_verification.json](metrics_verification.json): independent verification status.
- [verify_metrics.py](verify_metrics.py): reproducible independent check; run inside Slurm.
- [scheduling_compatibility.json](scheduling_compatibility.json): exact audited source migrations.

Bulky outputs and logs remain at `/scratch/yucz/dinov3_results/vits_claude_sweep_eval`.
