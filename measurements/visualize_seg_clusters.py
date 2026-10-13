"""Visualize ADE20K segmentation-derived per-image clusters for one image.

Loads a single ADE20K image + annotation, square-resizes both, majority-pools
the per-pixel segmentation map down to a patch grid, and saves a 2×2 figure:
the original image, the ground-truth segmentation, the per-patch clusters
(same color scheme as the GT, so labels match), and an α-blended overlay.

No model checkpoint is needed — the clusters come straight from the
annotations. Toggle which image to render by changing ``--image-index``.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dinov3.data.datasets.ade20k import ADE20K, _Split as ADE20KSplit
from measurements.compute_seg_anova import DEFAULT_ADE20K_ROOT, SegEvalTransform
from measurements.seg_anova import pool_segmentation_to_patches
from measurements.visualize_local_clusters import denormalize, make_distinct_colors


logger = logging.getLogger("measurements.visualize_seg_clusters")

IGNORE_COLOR = np.array([0.10, 0.10, 0.10])  # near-black for ignore_label patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-image visualization of ADE20K segmentation-derived clusters."
    )
    parser.add_argument("--ade20k-root", default=DEFAULT_ADE20K_ROOT)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--image-index", type=int, default=0,
                        help="Index into the ADE20K split. Change this to switch images.")
    parser.add_argument("--crop-size", type=int, default=448,
                        help="Image and mask are square-resized to this size before pooling.")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Edge length of the pooling window (matches model's patch size).")
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.55,
                        help="Overlay alpha for the cluster mask blended onto the image.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_label_color_table(unique_labels: np.ndarray, ignore_label: int) -> dict[int, np.ndarray]:
    """Assign a distinct color to each non-ignore label present in the image."""

    non_ignore = [int(l) for l in unique_labels.tolist() if int(l) != ignore_label]
    palette = make_distinct_colors(max(len(non_ignore), 1))
    return {lbl: palette[i] for i, lbl in enumerate(non_ignore)}


def labels_to_rgb(
    labels: np.ndarray,
    color_table: dict[int, np.ndarray],
    ignore_label: int,
) -> np.ndarray:
    """Map an int-label array to an RGB image using ``color_table``."""

    out = np.zeros((*labels.shape, 3), dtype=np.float64)
    for lbl in np.unique(labels):
        lbl_int = int(lbl)
        mask = labels == lbl
        if lbl_int == ignore_label or lbl_int not in color_table:
            out[mask] = IGNORE_COLOR
        else:
            out[mask] = color_table[lbl_int]
    return out


def default_output_path(split: str, image_index: int, crop: int, patch: int) -> Path:
    return (
        Path("measurements")
        / "seg_cluster_visualizations"
        / f"ade20k_{split}_idx{image_index}_crop{crop}_p{patch}.png"
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.crop_size % args.patch_size != 0:
        raise ValueError(
            f"--crop-size {args.crop_size} not divisible by --patch-size {args.patch_size}"
        )

    split_enum = ADE20KSplit.VAL if args.split == "val" else ADE20KSplit.TRAIN
    transform = SegEvalTransform(crop_size=args.crop_size)
    dataset = ADE20K(split=split_enum, root=args.ade20k_root, transforms=transform)
    if not (0 <= args.image_index < len(dataset)):
        raise IndexError(
            f"--image-index {args.image_index} out of range [0, {len(dataset)})"
        )

    logger.info("split=%s, dataset_size=%d, image_index=%d",
                args.split, len(dataset), args.image_index)

    image, mask = dataset[args.image_index]                              # (3, S, S), (S, S)
    img_np = denormalize(image)                                          # (S, S, 3)
    mask_np = mask.numpy()                                               # (S, S) int64

    # Pool to patch grid (same call as compute_seg_anova).
    pooled = pool_segmentation_to_patches(
        mask.unsqueeze(0), patch_size=args.patch_size, ignore_label=args.ignore_label
    )[0]                                                                 # (T,)
    Hp = args.crop_size // args.patch_size
    patch_grid = pooled.numpy().reshape(Hp, Hp)                          # (Hp, Hp)

    # Shared color table over all labels seen in this image (pixels and patches).
    all_labels = np.unique(np.concatenate([mask_np.ravel(), patch_grid.ravel()]))
    color_table = build_label_color_table(all_labels, args.ignore_label)
    K = len(color_table)
    logger.info("classes in image=%d (excluding ignore_label=%d)", K, args.ignore_label)

    gt_rgb = labels_to_rgb(mask_np, color_table, args.ignore_label)
    patch_rgb_small = labels_to_rgb(patch_grid, color_table, args.ignore_label)
    patch_rgb = np.repeat(
        np.repeat(patch_rgb_small, args.patch_size, axis=0), args.patch_size, axis=1
    )
    overlay = (1.0 - args.alpha) * img_np + args.alpha * patch_rgb

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"image (idx={args.image_index})")
    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("ground-truth segmentation (per-pixel)")
    axes[1, 0].imshow(patch_rgb)
    axes[1, 0].set_title(f"patch clusters ({Hp}×{Hp} grid, K={K})")
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"overlay (α={args.alpha:g})")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"ADE20K {args.split} · idx={args.image_index} · "
        f"crop={args.crop_size} · patch={args.patch_size} · ignore={args.ignore_label}"
    )
    fig.tight_layout()

    output = (
        args.output
        or default_output_path(args.split, args.image_index, args.crop_size, args.patch_size)
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
