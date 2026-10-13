from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Mapping

_MPL_CACHE_ROOT = Path(f"/tmp/dino_block_pca_plot_cache_{os.getuid()}")
_MPL_CONFIG_DIR = Path(os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_ROOT / "matplotlib")))
_XDG_CACHE_DIR = Path(os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_ROOT / "xdg")))
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from dinov3.data.transforms import CROP_DEFAULT_SIZE, RESIZE_DEFAULT_SIZE
from measurements.dino_measurement_utils import (
    build_balanced_batch,
    build_eval_dataset,
    extract_block_tokens,
    get_dataset_string,
    infer_config_file,
    infer_model_name,
    load_config,
    load_model_for_measurement,
    safe_output_stem,
)


logger = logging.getLogger("measurements.plot_block_pca")

mpl.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "mathtext.rm": "CMU Serif",
        "mathtext.it": "CMU Serif:italic",
        "mathtext.bf": "CMU Serif:bold",
        "font.family": "serif",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA projections of DINOv3 block outputs.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to teacher_checkpoint.pth.")
    parser.add_argument("--config-file", type=Path, default=None, help="Path to config.yaml.")
    parser.add_argument("--dataset", default=None, help="DINO dataset string. Defaults to train.dataset_path.")
    parser.add_argument("--classes", nargs="+", type=int, default=[0, 329, 658])
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("measurements") / "pca_plots")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)
    parser.add_argument("--clip-quantile", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def prepare_matplotlib_cache() -> None:
    _MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_pca_projection(x: torch.Tensor) -> torch.Tensor:
    """Project a token tensor of shape (B, T, C) to (B, T, 2) using the reference PCA definition."""

    if x.ndim != 3:
        raise ValueError(f"Expected tensor with shape (B, T, C), got {tuple(x.shape)}")
    x = x.float()
    batch_size, token_count, channel_count = x.shape
    if channel_count < 2:
        raise ValueError("PCA projection requires at least two channels.")

    x = x - x.mean(dim=(0, 1))
    _, _, vh = torch.linalg.svd(x.reshape(batch_size * token_count, channel_count), full_matrices=False)
    top2_eigenvec = vh[:2, :]
    return torch.matmul(x, top2_eigenvec.transpose(0, 1))


def _color_for_label(label: int, s: float = 0.65, v: float = 0.9):
    phi_conj = (5**0.5 - 1) / 2
    h = (label * phi_conj) % 1.0
    return mcolors.hsv_to_rgb((h, s, v))


def plot_sequences_by_label(
    projections: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_path: Path,
    title: str | None,
    color_map: Mapping[int, str | tuple] | None = None,
    clip_quantile: float | None = None,
    dpi: int = 150,
) -> None:
    if projections.ndim != 3 or projections.shape[-1] != 2:
        raise ValueError(f"Expected projections with shape (B, T, 2), got {tuple(projections.shape)}")
    if labels.ndim != 1 or labels.shape[0] != projections.shape[0]:
        raise ValueError(f"Expected labels with shape ({projections.shape[0]},), got {tuple(labels.shape)}")

    points = projections.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for class_id in np.unique(labels_np):
        mask = labels_np == class_id
        class_points = points[mask].reshape(-1, 2)
        if class_points.size == 0:
            continue
        color = color_map[int(class_id)] if color_map is not None and int(class_id) in color_map else _color_for_label(int(class_id))
        ax.scatter(class_points[:, 0], class_points[:, 1], color=color, s=10, alpha=0.8, marker="o", label=str(int(class_id)))

    if clip_quantile is not None:
        all_points = points.reshape(-1, 2)
        magnitudes = np.sqrt(all_points[:, 0] ** 2 + all_points[:, 1] ** 2)
        radius = np.quantile(magnitudes, clip_quantile)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_matplotlib_cache()

    checkpoint_path = args.checkpoint.resolve()
    config_file = (args.config_file or infer_config_file(checkpoint_path)).resolve()
    config = load_config(config_file)
    model_name = infer_model_name(config, config_file)
    dataset_string = get_dataset_string(config, args.dataset)

    logger.info("model_name=%s", model_name)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("config_file=%s", config_file)
    logger.info("dataset=%s", dataset_string)
    logger.info("classes=%s", args.classes)

    device = torch.device(args.device)
    dataset = build_eval_dataset(
        dataset_string=dataset_string,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
    )
    images, labels = build_balanced_batch(
        dataset,
        selected_classes=args.classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model, autocast_dtype = load_model_for_measurement(config, checkpoint_path, device)
    layer_tokens = extract_block_tokens(model, images, autocast_dtype=autocast_dtype)

    model_output_dir = args.output_dir / safe_output_stem(model_name, checkpoint_path)
    for layer_name, tokens in layer_tokens.items():
        logger.info("plotting %s from tokens %s", layer_name, tuple(tokens.shape))
        projections = compute_pca_projection(tokens)
        plot_sequences_by_label(
            projections,
            labels,
            output_path=model_output_dir / f"{layer_name}.pdf",
            title=f"{model_name}, {layer_name}",
            clip_quantile=args.clip_quantile,
            dpi=args.dpi,
        )

    logger.info("wrote PCA plots to %s", model_output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
