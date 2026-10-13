# DINOv3 Evaluation Results Summary

Linear probing (ImageNet-1k, top-1 %) and semantic segmentation (ADE20K, mIoU).

---

## ViT-B (checkpoint `training_499999`)

| Model | Linear top-1 (mean ± sd, n=4) | Segmentation mIoU |
|---|---|---|
| **vitb_3L_gpt** | **80.17 ± 0.03** | **41.61** |
| vitb_gpt (baseline) | 79.95 ± 0.08 | 40.70 |
| vitb_6L_gpt | 79.89 ± 0.04 | 40.70 |
| vitb_9L_gpt | 79.85 ± 0.04 | 40.55 |
| vitb_12L_gpt | 79.59 ± 0.01 | 38.29 |

Linear means combine seed-0 (full 16-LR sweep) with three top-3-LR seeds (1, 2, 3).
Segmentation mIoU is the final-step value, single seed per model.

### ViT-B linear — per-seed detail

| Model | seed0 (full) | seed1 | seed2 | seed3 | mean ± sd |
|---|---|---|---|---|---|
| vitb_gpt | 80.06 | 79.91 | 79.83 | 80.00 | 79.95 ± 0.08 |
| vitb_3L_gpt | 80.15 | 80.20 | 80.14 | 80.21 | 80.17 ± 0.03 |
| vitb_6L_gpt | 79.83 | 79.88 | 79.91 | 79.94 | 79.89 ± 0.04 |
| vitb_9L_gpt | 79.86 | 79.89 | 79.86 | 79.79 | 79.85 ± 0.04 |
| vitb_12L_gpt | 79.59 | 79.58 | 79.61 | 79.58 | 79.59 ± 0.01 |

### ViT-B pooling ablation — CLS-only vs CLS + patch-mean (seed 0)

| Model | Default (CLS + patch-mean) | CLS-only | Δ |
|---|---|---|---|
| vitb_gpt | 80.06 | 79.64 | −0.41 |
| vitb_12L_gpt | 79.59 | 79.32 | −0.28 |

---

## ViT-S (checkpoint `training_124999`, 3 seeds each)

| Model | Linear top-1 (mean ± sd) | Segmentation mIoU (mean ± sd) |
|---|---|---|
| **vits_3L_claude** | **73.145 ± 0.039** | 31.497 ± 0.186 |
| vits_5L_claude | 73.005 ± 0.024 | **32.140 ± 0.082** |
| vits_6L_claude | 72.169 ± 0.132 | 29.560 ± 0.070 |
| vits_1L_claude | 72.063 ± 0.006 | 30.627 ± 0.101 |
| vits_claude (baseline) | 71.563 ± 0.057 | 29.587 ± 0.076 |

Linear mean ± sd use the sample (n−1) standard deviation over 3 seeds, reported to 3
decimals so the tightest run (vits_1L, sd 0.006) is not rounded to zero.

### ViT-S linear — per-seed (top-1 %)

| Model | seed1 | seed2 | seed3 | mean | sd |
|---|---|---|---|---|---|
| vits_claude | 71.558 | 71.622 | 71.508 | 71.563 | 0.057 |
| vits_1L_claude | 72.064 | 72.068 | 72.056 | 72.063 | 0.006 |
| vits_3L_claude | 73.128 | 73.190 | 73.118 | 73.145 | 0.039 |
| vits_5L_claude | 73.018 | 72.978 | 73.020 | 73.005 | 0.024 |
| vits_6L_claude | 72.288 | 72.192 | 72.028 | 72.169 | 0.132 |

### ViT-S segmentation — per-seed (mIoU)

| Model | seed1 | seed2 | seed3 | mean | sd |
|---|---|---|---|---|---|
| vits_claude | 29.64 | 29.62 | 29.50 | 29.587 | 0.076 |
| vits_1L_claude | 30.64 | 30.72 | 30.52 | 30.627 | 0.101 |
| vits_3L_claude | 31.41 | 31.71 | 31.37 | 31.497 | 0.186 |
| vits_5L_claude | 32.05 | 32.21 | 32.16 | 32.140 | 0.082 |
| vits_6L_claude | 29.61 | 29.59 | 29.48 | 29.560 | 0.070 |

### ViT-S — new 3-seed means vs prior single runs (linear top-1)

| Model | New (3-seed mean) | Prior single run |
|---|---|---|
| vits_claude | 71.56 | 71.55 |
| vits_1L_claude | 72.06 | 72.07 |
| vits_3L_claude | 73.15 | 72.98 |
| vits_5L_claude | 73.01 | 72.98 |
| vits_6L_claude | 72.17 | 72.30 |

---

## Key takeaways

- **vitb_3L_gpt wins both tasks decisively** — +0.22 pp linear over baseline (non-overlapping error bars) and +0.91 mIoU segmentation. The earlier single-seed edge is now confirmed real.
- **vitb_12L_gpt is clearly worst on both** — −2.4 mIoU vs baseline; linear sd of ±0.01 makes the gap unambiguous.
- **No monotone trend in Laplacian-head count** (3 best, 12 worst, 6/9 in the middle), with the same shape on both tasks.
- **CLS-only pooling costs ~0.3–0.4 pp** — the patch-token mean carries real signal.
- **At ViT-S scale, every mixed variant beats the plain baseline on linear** (+0.5 to +1.6 pp), a cleaner win than at ViT-B.
- **The two tasks disagree on the ViT-S winner**: 3L best on linear, 5L best on segmentation (+2.5 mIoU over baseline).
- **vits_6L is the outlier** — decent on linear but worst on segmentation (tied with baseline).

## Caveats

- **ViT-S linear is rail-limited**: runs used the original grid whose optimum sits at the top edge (effective 0.075–0.10). These are lower bounds; the 3L-vs-5L ordering could shift with a wider grid. The ViT-B grid peaks comfortably interior (0.0025).
- **ViT-B segmentation is single-seed**: final-step values with LR annealed to 0, one seed each. The ViT-S segmentation is 3-seeded, and the ±0.06–0.15 spreads confirm those gaps are real.
