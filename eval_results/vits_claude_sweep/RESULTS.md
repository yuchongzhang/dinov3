# ViT-S training-seed sweep results

All 28 requested evaluations are complete and valid: two tasks for each of the
14 finished training runs. Results were refreshed on 2026-10-12 and independently
checked against raw result files. All 40 metric summaries match NumPy mean and
sample standard deviation (`ddof=1`) within absolute tolerance 1e-12.

Values below are percentages, reported as mean ± sample standard deviation
across training seeds. They are descriptive statistics, not confidence intervals.

| Model | Training seeds | n | ImageNet linear accuracy | ADE20K mIoU |
|---|---|---:|---:|---:|
| `vits_claude` | 0, 1, 2 | 3 | 72.08 ± 0.18 | 30.27 ± 0.70 |
| `vits_1L_claude` | 0, 1, 2 | 3 | 73.05 ± 0.27 | 31.05 ± 0.57 |
| `vits_3L_claude` | 0, 1, 2 | 3 | **73.34 ± 0.20** | **32.11 ± 0.54** |
| `vits_5L_claude` | 0, 1, 2 | 3 | 73.12 ± 0.13 | 31.83 ± 0.82 |
| `vits_6L_claude` | 0, 1 | 2 | 72.88 ± 0.02 | 29.76 ± 1.36 |

The 3L model has the highest mean on both tasks: +1.26 percentage points in
ImageNet accuracy and +1.85 points in ADE20K mIoU relative to `vits_claude`.
The 6L model has incomplete coverage because seed 2 did not finish training.
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

## Additional segmentation metrics

All metrics come from the same validation selected by highest mIoU. Every run's
best validation was at iteration 40000. Values remain percentages.

| Model | aAcc | Mean accuracy / recall | Dice | F-score | Precision |
|---|---:|---:|---:|---:|---:|
| `vits_claude` | 73.57 ± 0.44 | 39.18 ± 0.95 | 42.73 ± 0.80 | 43.41 ± 0.65 | 52.47 ± 0.73 |
| `vits_1L_claude` | 73.89 ± 0.34 | 40.11 ± 0.70 | 43.80 ± 0.67 | 44.19 ± 0.59 | 53.30 ± 0.74 |
| `vits_3L_claude` | 74.47 ± 0.36 | 41.34 ± 0.91 | 45.02 ± 0.73 | 45.63 ± 0.45 | 54.06 ± 0.24 |
| `vits_5L_claude` | 74.20 ± 0.45 | 40.92 ± 0.90 | 44.77 ± 0.99 | 45.17 ± 1.16 | 54.51 ± 1.11 |
| `vits_6L_claude` | 73.06 ± 0.69 | 38.42 ± 1.76 | 42.20 ± 1.65 | 42.77 ± 1.68 | 52.35 ± 0.82 |

## Protocol, exclusions, and artifacts

Every backbone uses its run's saved training configuration and final
`eval/training_142999/teacher_checkpoint.pth`. Linear probing uses evaluation seed
0 and segmentation uses evaluation seed 100, independently of training seed.
Linear probing completes 100100 iterations; segmentation completes 40000 and
selects the best mIoU from validations at 20000 and 40000. Saved evaluation
configurations and provenance match the requested protocols.

No requested evaluation is missing, failed, malformed, incompatible, or duplicated
in the accepted results. Two cancelled early linear pilot attempts and one failed
early segmentation pilot attempt remain excluded from statistics. Both tasks for
`vits_6L_claude_seed2` remain excluded because training was unfinished; that run
was left untouched. See [excluded.json](excluded.json) for the attempt history.

- [summary.csv](summary.csv): all 40 metric summaries, full precision, contributing seeds and coverage.
- [summary.json](summary.json): summaries and all 28 accepted evaluations with provenance.
- [per_run.csv](per_run.csv): per-run metrics, selected classifiers, validation steps and checkpoint paths.
- [inventory.json](inventory.json) and [inventory.csv](inventory.csv): all 15 training runs and evaluation job histories.
- [metrics_verification.json](metrics_verification.json): independent verification, CPU Slurm job 1592545.
- [verify_metrics.py](verify_metrics.py): reproducible independent check; run inside Slurm.

Bulky outputs and logs remain at
`/scratch/yucz/dinov3_results/vits_claude_sweep_eval`.
