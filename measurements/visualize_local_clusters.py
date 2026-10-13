"""Visualize per-image Ward clustering of DINOv3 last-layer patch tokens.

Picks one image from the (val) dataset, runs the same Ward agglomerative
clustering used in `measurements.compute_nested_anova` on its
L2-normalized last-layer patch tokens, and saves a side-by-side PNG with
each patch colored by its cluster label.

Toggle which image to render by changing ``--image-index``.
"""

from __future__ import annotations

import argparse
import colorsys
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from dinov3.data import make_dataset
from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
)
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval
from measurements.compute_anova import (
    extract_last_layer_tokens,
    infer_checkpoint_step,
    infer_config_file,
    infer_model_name,
    load_config,
)
from measurements.compute_nested_anova import get_dataset_string
from measurements.nested_anova import l2_normalize_tokens


logger = logging.getLogger("measurements.visualize_local_clusters")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-image local-cluster visualization of DINOv3 patch tokens."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset string. Defaults to train.dataset_path with TRAIN→VAL substitution.",
    )
    parser.add_argument("--use-train-dataset", action="store_true")
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Index into the dataset. Change this to switch which image is visualized.",
    )
    parser.add_argument("--local-distance-threshold", type=float, default=1.25)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay alpha for the cluster mask blended onto the image.",
    )
    return parser.parse_args()


def make_distinct_colors(k: int) -> np.ndarray:
    """Return ``(k, 3)`` RGB array of perceptually distinct colors."""

    if k <= 20:
        cmap = plt.get_cmap("tab20")
        return np.stack([np.array(cmap(i % 20))[:3] for i in range(k)], axis=0)
    # Golden-ratio hue stepping — distinct for arbitrary K.
    phi = (1 + 5 ** 0.5) / 2
    colors = np.zeros((k, 3))
    for i in range(k):
        h = (i / phi) % 1.0
        s = 0.85 if (i % 2 == 0) else 0.65
        v = 0.95 if (i % 3 != 0) else 0.75
        colors[i] = colorsys.hsv_to_rgb(h, s, v)
    return colors


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization on (3, H, W) → (H, W, 3) float in [0, 1]."""

    img = tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def cluster_patch_labels(tokens_TD: torch.Tensor, threshold: float) -> np.ndarray:
    """Return per-token int labels from Ward agglomerative clustering."""

    x_np = tokens_TD.detach().cpu().to(torch.float64).numpy()
    if x_np.shape[0] == 1:
        return np.zeros(1, dtype=np.int64)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="ward",
        metric="euclidean",
    )
    return clusterer.fit_predict(x_np).astype(np.int64)


def default_output_path(model_name: str, image_index: int, threshold: float) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    thr = f"{threshold:g}".replace(".", "p")
    return Path("measurements") / "cluster_visualizations" / f"{safe}_idx{image_index}_thr{thr}.png"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint_path = args.checkpoint.resolve()
    config_file = (args.config_file or infer_config_file(checkpoint_path)).resolve()
    config = load_config(config_file)
    dataset_str = get_dataset_string(config, args.dataset, use_train=args.use_train_dataset)
    model_name = infer_model_name(config, config_file)

    step = infer_checkpoint_step(checkpoint_path)
    logger.info("model=%s, step=%s, dataset=%s", model_name, step, dataset_str)
    logger.info("image_index=%d, threshold=%g", args.image_index, args.local_distance_threshold)

    device = torch.device(args.device)
    model = build_model_for_eval(config, checkpoint_path)
    model.to(device).eval()
    autocast_dtype = get_autocast_dtype(config)

    transform = make_classification_eval_transform(
        resize_size=args.resize_size, crop_size=args.crop_size
    )
    dataset = make_dataset(dataset_str=dataset_str, transform=transform)
    if not (0 <= args.image_index < len(dataset)):
        raise IndexError(
            f"--image-index {args.image_index} is out of range [0, {len(dataset)})"
        )

    image, target = dataset[args.image_index]                            # (3, H, W)
    image_batched = image.unsqueeze(0).to(device, non_blocking=True)     # (1, 3, H, W)

    with torch.inference_mode():
        tokens = extract_last_layer_tokens(
            model, image_batched, autocast_dtype, exclude_cls=True
        )
    tokens = l2_normalize_tokens(tokens)                                 # (1, T, D)
    T = int(tokens.shape[1])
    H_p = int(round(T ** 0.5))
    if H_p * H_p != T:
        raise RuntimeError(
            f"Token count {T} is not a perfect square; cannot reshape to a patch grid. "
            "Check that --crop-size matches the model's expected input."
        )
    if args.crop_size % H_p != 0:
        raise RuntimeError(
            f"crop-size {args.crop_size} is not divisible by the patch grid edge {H_p}."
        )
    patch_size = args.crop_size // H_p

    labels = cluster_patch_labels(tokens[0], args.local_distance_threshold)
    K = int(labels.max() + 1) if labels.size else 0
    sizes = np.bincount(labels, minlength=K)
    logger.info("K=%d clusters (sizes min=%d, max=%d, mean=%.1f)",
                K, sizes.min(), sizes.max(), sizes.mean())

    colors = make_distinct_colors(K)                                     # (K, 3)
    label_grid = labels.reshape(H_p, H_p)
    mask_small = colors[label_grid]                                      # (H_p, H_p, 3)
    mask = np.repeat(np.repeat(mask_small, patch_size, axis=0), patch_size, axis=1)

    img_np = denormalize(image)                                          # (H, W, 3)
    overlay = (1.0 - args.alpha) * img_np + args.alpha * mask

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title(f"image idx={args.image_index} (target={int(target)})")
    axes[1].imshow(mask)
    axes[1].set_title(f"clusters (K={K}, threshold={args.local_distance_threshold:g})")
    axes[2].imshow(overlay)
    axes[2].set_title(f"overlay (α={args.alpha:g})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{model_name}  ·  step={step}  ·  T={T} patches, patch_size={patch_size}")
    fig.tight_layout()

    output = (args.output or default_output_path(
        model_name, args.image_index, args.local_distance_threshold
    )).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
