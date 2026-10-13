from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinov3.data.transforms import CROP_DEFAULT_SIZE, RESIZE_DEFAULT_SIZE
from measurements.dino_measurement_utils import (
    build_eval_dataset,
    infer_model_name,
    load_config,
    load_model_for_measurement,
    make_simple_data_loader,
    safe_output_stem,
)
from measurements.spectrum_dino import (
    DEFAULT_MODEL_DIRS,
    extract_last_layer_tokens,
    get_validation_dataset_string,
    resolve_checkpoint_specs,
)


logger = logging.getLogger("measurements.image_spectrum_auc_dino")


@dataclass
class ImageSpectrumAUCStats:
    mean: float
    standard_error: float
    std: float
    num_images: int
    tokens_per_image: int
    embed_dim: int
    spectrum_length: int
    min_auc: float
    max_auc: float


def prepare_matplotlib_cache() -> None:
    cache_root = Path(f"/tmp/image_spectrum_auc_dino_plot_cache_{os.getuid()}")
    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib")))
    xdg_cache_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg")))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)


def short_model_label(model_name: str, checkpoint_path: Path) -> str:
    label = safe_output_stem(model_name, checkpoint_path)
    label = re.sub(r"_training_\d+$", "", label)
    label = re.sub(r"_claude$", "", label)
    return label


def unique_label(label: str, existing_labels: set[str]) -> str:
    if label not in existing_labels:
        return label
    suffix = 2
    while f"{label}_{suffix}" in existing_labels:
        suffix += 1
    return f"{label}_{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ImageNet-val per-image AUC of the cumulative normalized squared singular-value "
            "energy curve for DINOv3 last-layer patch-token features."
        ),
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        type=Path,
        default=[Path(model_dir) for model_dir in DEFAULT_MODEL_DIRS],
        help="Model run directories containing config.yaml and eval/training_*/teacher_checkpoint.pth.",
    )
    parser.add_argument("--checkpoint-step", type=int, default=124999)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit teacher checkpoints. Overrides --model-dirs and --checkpoint-step.",
    )
    parser.add_argument(
        "--config-files",
        nargs="+",
        type=Path,
        default=None,
        help="Config files matching --checkpoints. Defaults to config.yaml in each checkpoint's run directory.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset string. Defaults to train.dataset_path/train.dataset from the config with "
            "split=TRAIN replaced by split=VAL."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("measurements") / "dino_image_spectrum_auc_results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when shuffling ImageNet-val before --n-images sampling.",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=-1,
        help="Number of validation images to include. Use -1 for the full validation set.",
    )
    parser.add_argument(
        "--include-cls",
        dest="exclude_cls",
        action="store_false",
        help="Include the leading CLS token in X_i. By default only patch tokens are used.",
    )
    parser.set_defaults(exclude_cls=True)
    return parser.parse_args()


def normalized_squared_singular_value_energy(singular_values: torch.Tensor) -> torch.Tensor:
    if singular_values.ndim != 1:
        raise ValueError(f"Expected singular_values with shape (r,), got {tuple(singular_values.shape)}")
    if singular_values.numel() == 0:
        raise ValueError("Expected at least one singular value.")

    singular_values, _ = torch.sort(singular_values, descending=True)
    squared = singular_values.double().square()
    total_energy = squared.sum()
    if float(total_energy.item()) <= 0:
        raise ValueError("Cannot compute normalized energy for zero singular-value energy.")

    return (squared / total_energy).float()


def cumulative_energy_curve(singular_values: torch.Tensor) -> torch.Tensor:
    return normalized_squared_singular_value_energy(singular_values).cumsum(dim=0)


def singular_value_energy_auc(singular_values: torch.Tensor) -> torch.Tensor:
    return cumulative_energy_curve(singular_values).mean()


def batch_singular_value_energy_auc(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, C), got {tuple(tokens.shape)}")
    if tokens.shape[0] == 0 or tokens.shape[1] == 0 or tokens.shape[2] == 0:
        raise ValueError(f"Expected non-empty token tensor, got {tuple(tokens.shape)}")

    singular_values = torch.linalg.svdvals(tokens)
    singular_values, _ = torch.sort(singular_values, dim=1, descending=True)
    squared = singular_values.double().square()
    total_energy = squared.sum(dim=1, keepdim=True)
    if bool((total_energy <= 0).any().item()):
        raise ValueError("Cannot compute normalized energy for zero singular-value energy.")

    cumulative_energy = (squared / total_energy).cumsum(dim=1)
    return cumulative_energy.mean(dim=1).float()


@torch.inference_mode()
def compute_image_spectrum_auc_stats(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    n_images: int,
    exclude_cls: bool = True,
) -> ImageSpectrumAUCStats:
    sum_auc = 0.0
    sum_auc_squares = 0.0
    min_auc = math.inf
    max_auc = -math.inf
    num_images = 0
    tokens_per_image: int | None = None
    embed_dim: int | None = None
    spectrum_length: int | None = None

    for batch_idx, (samples, _targets) in enumerate(data_loader):
        if n_images > 0 and num_images >= n_images:
            break
        if n_images > 0 and num_images + samples.shape[0] > n_images:
            samples = samples[: n_images - num_images]

        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("batch %d / %d, images=%d", batch_idx, len(data_loader), num_images)

        samples = samples.to(device, non_blocking=True)
        tokens = extract_last_layer_tokens(
            model,
            samples,
            autocast_dtype=autocast_dtype,
            exclude_cls=exclude_cls,
        )
        batch_size, n_tokens, dim = tokens.shape

        if tokens_per_image is None:
            tokens_per_image = n_tokens
            embed_dim = dim
            spectrum_length = min(n_tokens, dim)
        elif n_tokens != tokens_per_image or dim != embed_dim:
            raise ValueError(
                f"Token shape changed from {(tokens_per_image, embed_dim)} to {(n_tokens, dim)}."
            )

        batch_auc = batch_singular_value_energy_auc(tokens).double().cpu()
        sum_auc += float(batch_auc.sum().item())
        sum_auc_squares += float(batch_auc.square().sum().item())
        min_auc = min(min_auc, float(batch_auc.min().item()))
        max_auc = max(max_auc, float(batch_auc.max().item()))
        num_images += batch_size

    if num_images == 0 or tokens_per_image is None or embed_dim is None or spectrum_length is None:
        raise RuntimeError("No images were processed.")

    mean = sum_auc / num_images
    if num_images > 1:
        variance = (sum_auc_squares - (sum_auc * sum_auc / num_images)) / (num_images - 1)
        variance = max(variance, 0.0)
    else:
        variance = 0.0
    std = math.sqrt(variance)
    standard_error = std / math.sqrt(num_images)

    return ImageSpectrumAUCStats(
        mean=mean,
        standard_error=standard_error,
        std=std,
        num_images=num_images,
        tokens_per_image=tokens_per_image,
        embed_dim=embed_dim,
        spectrum_length=spectrum_length,
        min_auc=min_auc,
        max_auc=max_auc,
    )


def run_for_checkpoint(
    *,
    checkpoint_path: Path,
    config_file: Path,
    dataset_override: str | None,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    resize_size: int,
    crop_size: int,
    log_every: int,
    seed: int,
    n_images: int,
    exclude_cls: bool,
) -> tuple[str, ImageSpectrumAUCStats, dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    config_file = config_file.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing config file: {config_file}")

    config = load_config(config_file)
    model_name = infer_model_name(config, config_file)
    dataset_string = get_validation_dataset_string(config, dataset_override)

    logger.info("Loading %s from %s", model_name, checkpoint_path)
    logger.info("dataset=%s", dataset_string)
    model, autocast_dtype = load_model_for_measurement(config, checkpoint_path, device)

    dataset = build_eval_dataset(dataset_string=dataset_string, resize_size=resize_size, crop_size=crop_size)
    data_loader = make_simple_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )

    stats = compute_image_spectrum_auc_stats(
        model=model,
        data_loader=data_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        log_every=log_every,
        n_images=n_images,
        exclude_cls=exclude_cls,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_stem = short_model_label(model_name, checkpoint_path)
    logger.info(
        "%s: images=%d, requested_n_images=%d, exclude_cls=%s, tokens/image=%d, embed_dim=%d, "
        "spectrum_length=%d, mean_auc=%.6f, standard_error=%.6g",
        model_name,
        stats.num_images,
        n_images,
        exclude_cls,
        stats.tokens_per_image,
        stats.embed_dim,
        stats.spectrum_length,
        stats.mean,
        stats.standard_error,
    )
    summary = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_string,
        "num_images": stats.num_images,
        "requested_n_images": n_images,
        "exclude_cls": exclude_cls,
        "tokens_per_image": stats.tokens_per_image,
        "embed_dim": stats.embed_dim,
        "spectrum_length": stats.spectrum_length,
        "mean_auc": stats.mean,
        "standard_error": stats.standard_error,
        "std": stats.std,
        "min_auc": stats.min_auc,
        "max_auc": stats.max_auc,
    }
    return output_stem, stats, summary


def save_auc_comparison_figure(
    summary: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    names = list(summary)
    means = [summary[name]["mean_auc"] for name in names]
    standard_errors = [summary[name]["standard_error"] for name in names]
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
    colors = [palette[i % len(palette)] for i in range(len(names))]

    fig_height = max(3.8, 0.46 * len(names) + 1.9)
    fig, ax = plt.subplots(figsize=(7.2, fig_height), dpi=dpi)
    positions = list(range(len(names)))
    ax.barh(
        positions,
        means,
        xerr=standard_errors,
        capsize=3,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333", "capthick": 1.0},
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean cumulative energy AUC")
    fig.suptitle("Per-image spectrum AUC", x=0.08, y=0.98, ha="left", fontsize=13, fontweight="bold")
    fig.text(
        0.08,
        0.925,
        "ImageNet validation, last-layer patch tokens. Error bars show standard error.",
        color="#555555",
        fontsize=9,
    )
    ax.grid(True, axis="x", color="#E3E6EA", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#B8BEC7")
    ax.tick_params(axis="y", length=0)
    value_offset = max(standard_errors, default=0.0) + 0.005
    for position, mean, standard_error in zip(positions, means, standard_errors):
        ax.text(
            mean + standard_error + value_offset,
            position,
            f"{mean:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )
    right_edge = min(1.0, max(mean + err for mean, err in zip(means, standard_errors)) + value_offset + 0.045)
    left_edge = max(0.0, min(means) - 0.04)
    if right_edge <= left_edge:
        right_edge = min(1.0, left_edge + 0.1)
    ax.set_xlim(left_edge, right_edge)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_matplotlib_cache()

    if args.n_images == 0 or args.n_images < -1:
        raise ValueError("--n-images must be -1 or a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    stats_by_model: dict[str, ImageSpectrumAUCStats] = {}
    summary: dict[str, dict[str, Any]] = {}
    for checkpoint_path, config_file in resolve_checkpoint_specs(args):
        name, stats, model_summary = run_for_checkpoint(
            checkpoint_path=checkpoint_path,
            config_file=config_file,
            dataset_override=args.dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            log_every=args.log_every,
            seed=args.seed,
            n_images=args.n_images,
            exclude_cls=args.exclude_cls,
        )
        name = unique_label(name, set(summary))
        model_summary["display_name"] = name
        stats_by_model[name] = stats
        summary[name] = model_summary

    if stats_by_model:
        save_auc_comparison_figure(
            summary,
            args.output_dir / "image_spectrum_auc_comparison.png",
            dpi=args.dpi,
        )
        with (args.output_dir / "image_spectrum_auc_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    logger.info("Wrote image spectrum AUC summary and comparison plot to %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
