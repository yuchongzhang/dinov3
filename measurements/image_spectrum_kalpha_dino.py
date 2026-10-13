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
    MODEL_COLOR_PALETTE,
    extract_last_layer_tokens,
    get_validation_dataset_string,
    model_color_map,
    resolve_checkpoint_specs,
)


logger = logging.getLogger("measurements.image_spectrum_kalpha_dino")

NEURIPS_2026_FONT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
    "font.size": 32,
    "axes.labelsize": 32,
    "axes.titlesize": 36,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 32,
    "mathtext.fontset": "stix",
}

K_ALPHA_PALETTE = list(MODEL_COLOR_PALETTE)


@dataclass
class ImageSpectrumKAlphaStats:
    mean: float
    standard_error: float
    std: float
    num_images: int
    tokens_per_image: int
    embed_dim: int
    spectrum_length: int
    min_k_alpha: int
    max_k_alpha: int


def prepare_matplotlib_cache() -> None:
    cache_root = Path(f"/tmp/image_spectrum_kalpha_dino_plot_cache_{os.getuid()}")
    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib")))
    xdg_cache_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg")))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)


def short_model_label(model_name: str, checkpoint_path: Path) -> str:
    label = safe_output_stem(model_name, checkpoint_path)
    label = re.sub(r"_training_\d+$", "", label)
    label = re.sub(r"_claude$", "", label)
    match = re.fullmatch(r"vits(?:_(\d+)L)?", label)
    if match:
        n_layers = match.group(1)
        return f"ViT-S-{n_layers}L" if n_layers else "ViT-S"
    return label


def unique_label(label: str, existing_labels: set[str]) -> str:
    if label not in existing_labels:
        return label
    suffix = 2
    while f"{label}_{suffix}" in existing_labels:
        suffix += 1
    return f"{label}_{suffix}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ImageNet-val per-image k_alpha for the cumulative normalized squared singular-value "
            "energy curve of DINOv3 last-layer patch-token features."
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Cumulative energy threshold in (0, 1).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("measurements") / "dino_image_spectrum_kalpha_results")
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
    parser.add_argument(
        "--center-per-image",
        action="store_true",
        help=(
            "Subtract each image's mean token vector before SVD. In the default patch-only mode, "
            "this subtracts the mean patch embedding independently for each image."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip model evaluation and regenerate plots from metadata/plot_data.json in the selected output directory.",
    )
    parser.set_defaults(exclude_cls=True)
    return parser.parse_args(argv)


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
    cumulative_energy = normalized_squared_singular_value_energy(singular_values).cumsum(dim=0)
    cumulative_energy[-1] = 1.0
    return cumulative_energy


def singular_value_energy_k_alpha(singular_values: torch.Tensor, *, alpha: float) -> torch.Tensor:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    cumulative_energy = cumulative_energy_curve(singular_values)
    return torch.searchsorted(cumulative_energy, torch.tensor(alpha, device=cumulative_energy.device)).add(1).long()


def center_tokens_per_image(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, C), got {tuple(tokens.shape)}")
    if tokens.shape[1] == 0:
        raise ValueError(f"Expected at least one token per image, got {tuple(tokens.shape)}")

    return tokens - tokens.mean(dim=1, keepdim=True)


def batch_singular_value_energy_k_alpha(
    tokens: torch.Tensor,
    *,
    alpha: float,
    center_per_image: bool = False,
) -> torch.Tensor:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, C), got {tuple(tokens.shape)}")
    if tokens.shape[0] == 0 or tokens.shape[1] == 0 or tokens.shape[2] == 0:
        raise ValueError(f"Expected non-empty token tensor, got {tuple(tokens.shape)}")

    if center_per_image:
        tokens = center_tokens_per_image(tokens)

    singular_values = torch.linalg.svdvals(tokens)
    singular_values, _ = torch.sort(singular_values, dim=1, descending=True)
    squared = singular_values.double().square()
    total_energy = squared.sum(dim=1, keepdim=True)
    if bool((total_energy <= 0).any().item()):
        raise ValueError("Cannot compute normalized energy for zero singular-value energy.")

    cumulative_energy = (squared / total_energy).cumsum(dim=1)
    cumulative_energy[:, -1] = 1.0
    return cumulative_energy.ge(alpha).long().argmax(dim=1).add(1)


@torch.inference_mode()
def compute_image_spectrum_k_alpha_stats(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    n_images: int,
    alpha: float,
    exclude_cls: bool = True,
    center_per_image: bool = False,
) -> ImageSpectrumKAlphaStats:
    sum_k = 0.0
    sum_k_squares = 0.0
    min_k = math.inf
    max_k = -math.inf
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

        batch_k = batch_singular_value_energy_k_alpha(
            tokens,
            alpha=alpha,
            center_per_image=center_per_image,
        ).double().cpu()
        sum_k += float(batch_k.sum().item())
        sum_k_squares += float(batch_k.square().sum().item())
        min_k = min(min_k, int(batch_k.min().item()))
        max_k = max(max_k, int(batch_k.max().item()))
        num_images += batch_size

    if num_images == 0 or tokens_per_image is None or embed_dim is None or spectrum_length is None:
        raise RuntimeError("No images were processed.")

    mean = sum_k / num_images
    if num_images > 1:
        variance = (sum_k_squares - (sum_k * sum_k / num_images)) / (num_images - 1)
        variance = max(variance, 0.0)
    else:
        variance = 0.0
    std = math.sqrt(variance)
    standard_error = std / math.sqrt(num_images)

    return ImageSpectrumKAlphaStats(
        mean=mean,
        standard_error=standard_error,
        std=std,
        num_images=num_images,
        tokens_per_image=tokens_per_image,
        embed_dim=embed_dim,
        spectrum_length=spectrum_length,
        min_k_alpha=int(min_k),
        max_k_alpha=int(max_k),
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
    alpha: float,
    exclude_cls: bool,
    center_per_image: bool,
) -> tuple[str, ImageSpectrumKAlphaStats, dict[str, Any]]:
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

    stats = compute_image_spectrum_k_alpha_stats(
        model=model,
        data_loader=data_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        log_every=log_every,
        n_images=n_images,
        alpha=alpha,
        exclude_cls=exclude_cls,
        center_per_image=center_per_image,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_stem = short_model_label(model_name, checkpoint_path)
    logger.info(
        "%s: images=%d, requested_n_images=%d, alpha=%.4f, exclude_cls=%s, center_per_image=%s, "
        "tokens/image=%d, embed_dim=%d, "
        "spectrum_length=%d, mean_k_alpha=%.6f, standard_error=%.6g",
        model_name,
        stats.num_images,
        n_images,
        alpha,
        exclude_cls,
        center_per_image,
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
        "alpha": alpha,
        "exclude_cls": exclude_cls,
        "center_per_image": center_per_image,
        "tokens_per_image": stats.tokens_per_image,
        "embed_dim": stats.embed_dim,
        "spectrum_length": stats.spectrum_length,
        "mean_k_alpha": stats.mean,
        "standard_error": stats.standard_error,
        "std": stats.std,
        "min_k_alpha": stats.min_k_alpha,
        "max_k_alpha": stats.max_k_alpha,
    }
    return output_stem, stats, summary


def save_k_alpha_comparison_figure(
    summary: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    alpha: float,
    dpi: int,
    color_mode: str,
    exclude_cls: bool,
    center_per_image: bool,
) -> None:
    import matplotlib.pyplot as plt

    with plt.rc_context(NEURIPS_2026_FONT_RC):
        names = list(summary)
        means = [summary[name]["mean_k_alpha"] for name in names]
        standard_errors = [summary[name]["standard_error"] for name in names]
        color_by_name = model_color_map(names)
        if color_mode == "blue":
            colors = [K_ALPHA_PALETTE[0]] * len(names)
        elif color_mode == "multicolor":
            colors = [color_by_name[name] for name in names]
        else:
            raise ValueError(f"Unsupported color_mode: {color_mode}")

        fig_width = max(14.0, 1.4 * len(names) + 5.0)
        fig, ax = plt.subplots(figsize=(fig_width, 12.5), dpi=dpi)
        spacing = 0.16
        positions = [spacing * idx for idx in range(len(names))]
        ax.bar(
            positions,
            means,
            width=0.10,
            yerr=standard_errors,
            capsize=3,
            color=colors,
            edgecolor="white",
            linewidth=1.0,
            error_kw={"elinewidth": 1.0, "ecolor": "#333333", "capthick": 1.0},
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(names)
        if positions:
            ax.set_xlim(positions[0] - 0.10, positions[-1] + 0.10)
        # ax.set_ylabel(rf"Mean $k_{{\alpha}}$ at $\alpha = {alpha:g}$")
        ax.set_ylabel(rf"Number of directions to explain {100*alpha:g}% of variance")
        ax.grid(True, axis="y", color="#E3E6EA", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#B8BEC7")
        ax.spines["bottom"].set_color("#B8BEC7")
        ax.tick_params(axis="x", length=0)
        value_offset = max(standard_errors, default=0.0) + 0.6
        for position, mean, standard_error in zip(positions, means, standard_errors):
            ax.text(
                position,
                mean + standard_error + value_offset,
                f"{mean:.1f}",
                va="bottom",
                ha="center",
                fontsize=28,
                color="#333333",
            )
        lower_values = [mean - err for mean, err in zip(means, standard_errors)]
        upper_values = [mean + err for mean, err in zip(means, standard_errors)]
        y_min = min(lower_values)
        y_max = max(upper_values)
        spread = max(y_max - y_min, 1.0)
        bottom = max(0.0, y_min - 0.35 * spread)
        top = y_max + 0.65 * spread + value_offset
        if top <= bottom:
            top = bottom + 1.0
        ax.set_ylim(bottom, top)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def save_k_alpha_plot_data(
    summary: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    alpha: float,
    exclude_cls: bool,
    center_per_image: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(summary)
    spacing = 0.16
    color_by_name = model_color_map(names)
    plot_data = {
        "model_order": names,
        "settings": {
            "alpha": alpha,
            "exclude_cls": exclude_cls,
            "center_per_image": center_per_image,
            "bar_spacing": spacing,
            "palette": K_ALPHA_PALETTE,
            "colors_by_mode": {
                "blue": [K_ALPHA_PALETTE[0] for _name in names],
                "multicolor": [color_by_name[name] for name in names],
            },
        },
        "bars": [
            {
                "label": name,
                "position": spacing * idx,
                "mean_k_alpha": summary[name]["mean_k_alpha"],
                "standard_error": summary[name]["standard_error"],
                "std": summary[name]["std"],
                "num_images": summary[name]["num_images"],
                "min_k_alpha": summary[name]["min_k_alpha"],
                "max_k_alpha": summary[name]["max_k_alpha"],
            }
            for idx, name in enumerate(names)
        ],
        "summary": summary,
    }
    with output_path.open("w") as f:
        json.dump(plot_data, f, indent=2)
        f.write("\n")


def load_k_alpha_plot_data(input_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing saved plot data: {input_path}")

    with input_path.open() as f:
        plot_data = json.load(f)

    summary = plot_data.get("summary")
    settings = plot_data.get("settings", {})
    if not isinstance(summary, dict) or not isinstance(settings, dict):
        raise ValueError(f"Invalid k-alpha plot data in {input_path}")

    model_order = plot_data.get("model_order", list(summary))
    return {name: summary[name] for name in model_order}, dict(settings)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_matplotlib_cache()

    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1).")
    if args.n_images == 0 or args.n_images < -1:
        raise ValueError("--n-images must be -1 or a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    alpha_suffix = f"alpha={args.alpha:g}"
    result_suffix = f"{alpha_suffix}_centered" if args.center_per_image else alpha_suffix
    output_dir = args.output_dir / result_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plot_data_path = output_dir / "metadata" / "plot_data.json"
        summary, settings = load_k_alpha_plot_data(plot_data_path)
        plot_alpha = float(settings.get("alpha", args.alpha))
        plot_exclude_cls = bool(settings.get("exclude_cls", args.exclude_cls))
        plot_center_per_image = bool(settings.get("center_per_image", args.center_per_image))
        save_k_alpha_comparison_figure(
            summary,
            output_dir / f"image_spectrum_{result_suffix}_comparison_multicolor.png",
            alpha=plot_alpha,
            dpi=args.dpi,
            color_mode="multicolor",
            exclude_cls=plot_exclude_cls,
            center_per_image=plot_center_per_image,
        )
        save_k_alpha_comparison_figure(
            summary,
            output_dir / f"image_spectrum_{result_suffix}_comparison_blue.png",
            alpha=plot_alpha,
            dpi=args.dpi,
            color_mode="blue",
            exclude_cls=plot_exclude_cls,
            center_per_image=plot_center_per_image,
        )
        logger.info("Regenerated image spectrum k-alpha plots from %s", plot_data_path.resolve())
        return 0

    device = torch.device(args.device)

    stats_by_model: dict[str, ImageSpectrumKAlphaStats] = {}
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
            alpha=args.alpha,
            exclude_cls=args.exclude_cls,
            center_per_image=args.center_per_image,
        )
        name = unique_label(name, set(summary))
        model_summary["display_name"] = name
        stats_by_model[name] = stats
        summary[name] = model_summary

    if stats_by_model:
        save_k_alpha_comparison_figure(
            summary,
            output_dir / f"image_spectrum_{result_suffix}_comparison_multicolor.png",
            alpha=args.alpha,
            dpi=args.dpi,
            color_mode="multicolor",
            exclude_cls=args.exclude_cls,
            center_per_image=args.center_per_image,
        )
        save_k_alpha_comparison_figure(
            summary,
            output_dir / f"image_spectrum_{result_suffix}_comparison_blue.png",
            alpha=args.alpha,
            dpi=args.dpi,
            color_mode="blue",
            exclude_cls=args.exclude_cls,
            center_per_image=args.center_per_image,
        )
        with (output_dir / f"image_spectrum_{result_suffix}_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        metadata_dir = output_dir / "metadata"
        save_k_alpha_plot_data(
            summary,
            metadata_dir / "plot_data.json",
            alpha=args.alpha,
            exclude_cls=args.exclude_cls,
            center_per_image=args.center_per_image,
        )

    logger.info("Wrote image spectrum k_alpha summary and comparison plot to %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
