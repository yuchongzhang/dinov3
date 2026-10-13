from __future__ import annotations

import argparse
import json
import logging
import os
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
)
from measurements.spectrum_dino import (
    DEFAULT_MODEL_DIRS,
    NORMALIZATION_MODES,
    PLOT_FONT_RC,
    PLOT_WINDOWS,
    SPECTRUM_LINEWIDTH,
    SUBSET_COMPARISON_LABELS,
    cutoff_length_for_spectrum,
    extract_last_layer_tokens,
    get_validation_dataset_string,
    model_color_map,
    output_mode_name,
    resolve_checkpoint_specs,
    safe_plot_stem,
    save_figure_png_and_pdf,
    short_model_label,
    unique_label,
)


logger = logging.getLogger("measurements.image_spectrum_dino")

IMAGE_SPECTRUM_FONT_RC = {
    **PLOT_FONT_RC,
    "font.size": 32,
    "axes.labelsize": 32,
    "axes.titlesize": 36,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 32,
}


@dataclass
class ImageSpectrumStats:
    mean: torch.Tensor
    std: torch.Tensor
    num_images: int
    tokens_per_image: int
    embed_dim: int
    spectrum_length: int


def prepare_matplotlib_cache() -> None:
    cache_root = Path(f"/tmp/image_spectrum_dino_plot_cache_{os.getuid()}")
    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib")))
    xdg_cache_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg")))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ImageNet-val per-image eigenvalue spectra of X_i^T X_i for "
            "DINOv3 last-layer token features, then plot mean +/- standard deviation."
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
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated compatibility flag. Both normalized and unnormalized plots are always saved.",
    )
    parser.add_argument("--y-scale", choices=("linear", "log"), default="log")
    parser.add_argument("--x-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--output-dir", type=Path, default=Path("measurements") / "dino_image_spectrum_results")
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
        "--top-k-eig",
        "--topk-eig",
        dest="top_k_eig",
        type=int,
        default=-1,
        help="Only compute and visualize the top k eigenvalues by index. Use -1 for all C eigenvalues.",
    )
    cls_group = parser.add_mutually_exclusive_group()
    cls_group.add_argument(
        "--include-cls",
        dest="exclude_cls",
        action="store_false",
        help="Include the leading CLS token in X_i. By default only patch tokens are used.",
    )
    cls_group.add_argument(
        "--exclude-cls",
        dest="exclude_cls",
        action="store_true",
        help="Exclude the leading CLS token and use patch tokens only.",
    )
    parser.set_defaults(exclude_cls=True)
    parser.add_argument(
        "--center-per-image",
        action="store_true",
        help="Subtract each image's mean selected-token embedding before computing its spectrum.",
    )
    parser.add_argument(
        "--save-values",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated compatibility flag. JSON summaries are always saved and .pt files are never saved.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip model evaluation and regenerate plots from metadata/plot_data.json in the selected output directory.",
    )
    return parser.parse_args(argv)


def spectrum_length_for_plot(embed_dim: int, top_k_eig: int) -> int:
    if top_k_eig > 0:
        return min(top_k_eig, embed_dim)
    return embed_dim


def center_tokens_per_image(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, C), got {tuple(tokens.shape)}")
    if tokens.shape[1] == 0:
        raise ValueError(f"Expected at least one token per image, got {tuple(tokens.shape)}")
    return tokens - tokens.mean(dim=1, keepdim=True)


def image_gram_eigenvalues(
    tokens: torch.Tensor,
    *,
    spectrum_length: int,
    center_per_image: bool = False,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, C), got {tuple(tokens.shape)}")
    if spectrum_length <= 0 or spectrum_length > tokens.shape[-1]:
        raise ValueError(
            f"spectrum_length must be in [1, embed_dim={tokens.shape[-1]}], got {spectrum_length}."
        )

    if center_per_image:
        tokens = center_tokens_per_image(tokens)

    eigvals = torch.linalg.svdvals(tokens).square()
    eigvals, _ = torch.sort(eigvals, dim=1, descending=True)
    eigvals = eigvals[:, :spectrum_length]
    if eigvals.shape[1] < spectrum_length:
        eigvals = torch.nn.functional.pad(eigvals, (0, spectrum_length - eigvals.shape[1]))
    return eigvals.float()


@torch.inference_mode()
def compute_image_spectrum_stats(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    n_images: int,
    top_k_eig: int,
    exclude_cls: bool = True,
    center_per_image: bool = False,
) -> ImageSpectrumStats:
    sum_values: torch.Tensor | None = None
    sum_squares: torch.Tensor | None = None
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
            spectrum_length = spectrum_length_for_plot(dim, top_k_eig)
        elif n_tokens != tokens_per_image or dim != embed_dim:
            raise ValueError(
                f"Token shape changed from {(tokens_per_image, embed_dim)} to {(n_tokens, dim)}."
            )

        assert spectrum_length is not None
        batch_spectra = image_gram_eigenvalues(
            tokens,
            spectrum_length=spectrum_length,
            center_per_image=center_per_image,
        ).double().cpu()
        batch_sum = batch_spectra.sum(dim=0)
        batch_sum_squares = batch_spectra.square().sum(dim=0)
        sum_values = batch_sum if sum_values is None else sum_values + batch_sum
        sum_squares = batch_sum_squares if sum_squares is None else sum_squares + batch_sum_squares
        num_images += batch_size

    if num_images == 0 or tokens_per_image is None or embed_dim is None or spectrum_length is None:
        raise RuntimeError("No images were processed.")

    assert sum_values is not None and sum_squares is not None
    mean = sum_values / num_images
    variance = (sum_squares / num_images) - mean.square()
    std = variance.clamp_min(0).sqrt()
    return ImageSpectrumStats(
        mean=mean.float(),
        std=std.float(),
        num_images=num_images,
        tokens_per_image=tokens_per_image,
        embed_dim=embed_dim,
        spectrum_length=spectrum_length,
    )


def _values_for_plot(
    stats: ImageSpectrumStats,
    *,
    normalize: bool,
    y_scale: str,
    top_k_eig: int,
    plot_window: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = stats.mean.clone()
    std = stats.std.clone()
    if top_k_eig > 0:
        length = min(top_k_eig, mean.numel())
        mean = mean[:length]
        std = std[:length]
    if plot_window == "cutoff":
        length = cutoff_length_for_spectrum(mean)
        mean = mean[:length]
        std = std[:length]
    elif plot_window != "full":
        raise ValueError(f"Unsupported plot_window: {plot_window}")

    if normalize:
        scale = mean.max().clamp_min(1e-30)
        mean = mean / scale
        std = std / scale

    lower = (mean - std).clamp_min(0)
    upper = (mean + std).clamp_min(0)
    if y_scale == "log":
        positive = torch.cat((mean[mean > 0], upper[upper > 0]))
        floor = positive.min() * 1e-6 if positive.numel() else torch.tensor(1e-30)
        floor = floor.clamp_min(1e-30)
        mean = mean.clamp_min(floor)
        lower = lower.clamp_min(floor)
        upper = upper.clamp_min(floor)
    return mean, lower, upper


def save_single_image_spectrum_figure(
    stats: ImageSpectrumStats,
    model_name: str,
    output_path: Path,
    *,
    x_scale: str,
    y_scale: str,
    normalize: bool,
    top_k_eig: int,
    plot_window: str,
    dpi: int,
    center_per_image: bool,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean, lower, upper = _values_for_plot(
        stats,
        normalize=normalize,
        y_scale=y_scale,
        top_k_eig=top_k_eig,
        plot_window=plot_window,
    )
    indices = torch.arange(1, mean.numel() + 1)

    with plt.rc_context(IMAGE_SPECTRUM_FONT_RC):
        fig, ax = plt.subplots(figsize=(10, 9.5), dpi=dpi)
        ax.plot(indices.numpy(), mean.numpy(), marker="", linewidth=SPECTRUM_LINEWIDTH)
        ax.fill_between(indices.numpy(), lower.numpy(), upper.numpy(), alpha=0.18, linewidth=0)
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlabel("Eigenvalue index")
        ylabel = "Mean eigenvalue of $X_i^\\top X_i$"
        if center_per_image:
            ylabel = "Mean eigenvalue of $\\tilde X_i^\\top \\tilde X_i$"
        if normalize:
            ylabel = f"Normalized {ylabel[0].lower()}{ylabel[1:]}"
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save_figure_png_and_pdf(fig, output_path)
        plt.close(fig)


def save_combined_image_spectrum_figure(
    stats_by_model: dict[str, ImageSpectrumStats],
    output_path: Path,
    *,
    x_scale: str,
    y_scale: str,
    normalize: bool,
    top_k_eig: int,
    plot_window: str,
    dpi: int,
    center_per_image: bool,
    title: str,
    color_by_name: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(IMAGE_SPECTRUM_FONT_RC):
        fig, ax = plt.subplots(figsize=(12, 10.5), dpi=dpi)
        for name, stats in stats_by_model.items():
            mean, lower, upper = _values_for_plot(
                stats,
                normalize=normalize,
                y_scale=y_scale,
                top_k_eig=top_k_eig,
                plot_window=plot_window,
            )
            indices = torch.arange(1, mean.numel() + 1)
            ax.plot(
                indices.numpy(),
                mean.numpy(),
                label=name,
                color=color_by_name[name],
                linewidth=SPECTRUM_LINEWIDTH,
            )
            ax.fill_between(
                indices.numpy(),
                lower.numpy(),
                upper.numpy(),
                color=color_by_name[name],
                alpha=0.13,
                linewidth=0,
            )

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlabel("Eigenvalue index")
        ylabel = "Mean eigenvalue of $X_i^\\top X_i$"
        if center_per_image:
            ylabel = "Mean eigenvalue of $\\tilde X_i^\\top \\tilde X_i$"
        if normalize:
            ylabel = f"Normalized {ylabel[0].lower()}{ylabel[1:]}"
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=32)
        fig.tight_layout()
        save_figure_png_and_pdf(fig, output_path)
        plt.close(fig)


def save_all_image_spectrum_plots(
    stats_by_model: dict[str, ImageSpectrumStats],
    output_dir: Path,
    *,
    x_scale: str,
    y_scale: str,
    top_k_eig: int,
    dpi: int,
    exclude_cls: bool,
    center_per_image: bool,
) -> None:
    color_by_name = model_color_map(stats_by_model)
    for name, stats in stats_by_model.items():
        for plot_window in PLOT_WINDOWS:
            for normalize, normalize_dir in NORMALIZATION_MODES:
                save_single_image_spectrum_figure(
                    stats,
                    name,
                    output_dir / "individual" / plot_window / normalize_dir / f"{safe_plot_stem(name)}.png",
                    x_scale=x_scale,
                    y_scale=y_scale,
                    normalize=normalize,
                    top_k_eig=top_k_eig,
                    plot_window=plot_window,
                    dpi=dpi,
                    center_per_image=center_per_image,
                )

    combined_specs = [("all_models", stats_by_model, "ImageNet-val per-image spectrum across models")]
    subset = {label: stats_by_model[label] for label in SUBSET_COMPARISON_LABELS if label in stats_by_model}
    if len(subset) == len(SUBSET_COMPARISON_LABELS):
        combined_specs.append(
            (
                "ViT-S_vs_ViT-S-5L",
                subset,
                "ImageNet-val per-image spectrum: ViT-S vs ViT-S-5L",
            )
        )
    else:
        missing = [label for label in SUBSET_COMPARISON_LABELS if label not in stats_by_model]
        logger.warning(
            "Skipping ViT-S vs ViT-S-5L image spectrum plot because labels are missing: %s",
            ", ".join(missing),
        )

    for stem, group_stats, title in combined_specs:
        for plot_window in PLOT_WINDOWS:
            for normalize, normalize_dir in NORMALIZATION_MODES:
                save_combined_image_spectrum_figure(
                    group_stats,
                    output_dir / "combined" / plot_window / normalize_dir / f"{stem}.png",
                    x_scale=x_scale,
                    y_scale=y_scale,
                    normalize=normalize,
                    top_k_eig=top_k_eig,
                    plot_window=plot_window,
                    dpi=dpi,
                    center_per_image=center_per_image,
                    title=title,
                    color_by_name=color_by_name,
                )


def save_image_spectrum_plot_data(
    stats_by_model: dict[str, ImageSpectrumStats],
    summary: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    x_scale: str,
    y_scale: str,
    top_k_eig: int,
    exclude_cls: bool,
    center_per_image: bool,
    resize_size: int,
    crop_size: int,
    n_images: int,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    color_by_name = model_color_map(stats_by_model)
    plot_data = {
        "model_order": list(stats_by_model),
        "colors": {name: color_by_name[name] for name in stats_by_model},
        "settings": {
            "x_scale": x_scale,
            "y_scale": y_scale,
            "top_k_eig": top_k_eig,
            "exclude_cls": exclude_cls,
            "center_per_image": center_per_image,
            "resize_size": resize_size,
            "crop_size": crop_size,
            "n_images": n_images,
            "seed": seed,
        },
        "models": {
            name: {
                "indices": list(range(1, stats.mean.numel() + 1)),
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "cutoff_eigenvalues": cutoff_length_for_spectrum(stats.mean),
                "summary": summary[name],
            }
            for name, stats in stats_by_model.items()
        },
    }
    with output_path.open("w") as f:
        json.dump(plot_data, f, indent=2)
        f.write("\n")


def load_image_spectrum_plot_data(
    input_path: Path,
) -> tuple[dict[str, ImageSpectrumStats], dict[str, dict[str, Any]], dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing saved plot data: {input_path}")

    with input_path.open() as f:
        plot_data = json.load(f)

    settings = dict(plot_data.get("settings", {}))
    models = plot_data.get("models")
    model_order = plot_data.get("model_order")
    if not isinstance(models, dict) or not isinstance(model_order, list):
        raise ValueError(f"Invalid image spectrum plot data in {input_path}")

    stats_by_model: dict[str, ImageSpectrumStats] = {}
    summary: dict[str, dict[str, Any]] = {}
    for name in model_order:
        model_data = models[name]
        model_summary = dict(model_data["summary"])
        mean = torch.tensor(model_data["mean"], dtype=torch.float32)
        std = torch.tensor(model_data["std"], dtype=torch.float32)
        stats_by_model[name] = ImageSpectrumStats(
            mean=mean,
            std=std,
            num_images=int(model_summary["num_images"]),
            tokens_per_image=int(model_summary["tokens_per_image"]),
            embed_dim=int(model_summary["embed_dim"]),
            spectrum_length=int(model_summary.get("plotted_eigenvalues", mean.numel())),
        )
        summary[name] = model_summary

    return stats_by_model, summary, settings


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
    top_k_eig: int,
    exclude_cls: bool,
    center_per_image: bool,
) -> tuple[str, ImageSpectrumStats, dict[str, Any]]:
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

    stats = compute_image_spectrum_stats(
        model=model,
        data_loader=data_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        log_every=log_every,
        n_images=n_images,
        top_k_eig=top_k_eig,
        exclude_cls=exclude_cls,
        center_per_image=center_per_image,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_stem = short_model_label(model_name, checkpoint_path)

    logger.info(
        "%s: images=%d, requested_n_images=%d, exclude_cls=%s, center_per_image=%s, tokens/image=%d, embed_dim=%d, "
        "plotted=%d, top_mean=%.4g, bottom_mean=%.4g",
        model_name,
        stats.num_images,
        n_images,
        exclude_cls,
        center_per_image,
        stats.tokens_per_image,
        stats.embed_dim,
        stats.spectrum_length,
        float(stats.mean[0]),
        float(stats.mean[-1]),
    )
    summary = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_string,
        "num_images": stats.num_images,
        "requested_n_images": n_images,
        "top_k_eig": top_k_eig,
        "exclude_cls": exclude_cls,
        "center_per_image": center_per_image,
        "tokens_per_image": stats.tokens_per_image,
        "embed_dim": stats.embed_dim,
        "rank": stats.embed_dim,
        "plotted_eigenvalues": stats.spectrum_length,
        "cutoff_eigenvalues": cutoff_length_for_spectrum(stats.mean),
        "top_mean": float(stats.mean[0]),
        "bottom_mean": float(stats.mean[-1]),
        "top_std": float(stats.std[0]),
        "bottom_std": float(stats.std[-1]),
    }
    return output_stem, stats, summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_matplotlib_cache()

    if args.n_images == 0 or args.n_images < -1:
        raise ValueError("--n-images must be -1 or a positive integer.")
    if args.top_k_eig == 0 or args.top_k_eig < -1:
        raise ValueError("--top-k-eig must be -1 or a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    center_suffix = "per_image_centered" if args.center_per_image else None
    output_dir = args.output_dir / output_mode_name(exclude_cls=args.exclude_cls, center_suffix=center_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plot_data_path = output_dir / "metadata" / "plot_data.json"
        stats_by_model, _summary, settings = load_image_spectrum_plot_data(plot_data_path)
        save_all_image_spectrum_plots(
            stats_by_model,
            output_dir,
            x_scale=str(settings.get("x_scale", args.x_scale)),
            y_scale=str(settings.get("y_scale", args.y_scale)),
            top_k_eig=int(settings.get("top_k_eig", args.top_k_eig)),
            dpi=args.dpi,
            exclude_cls=bool(settings.get("exclude_cls", args.exclude_cls)),
            center_per_image=bool(settings.get("center_per_image", args.center_per_image)),
        )
        logger.info("Regenerated image spectrum plots from %s", plot_data_path.resolve())
        return 0

    device = torch.device(args.device)

    stats_by_model: dict[str, ImageSpectrumStats] = {}
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
            top_k_eig=args.top_k_eig,
            exclude_cls=args.exclude_cls,
            center_per_image=args.center_per_image,
        )
        name = unique_label(name, set(stats_by_model))
        model_summary["display_name"] = name
        stats_by_model[name] = stats
        summary[name] = model_summary

    if stats_by_model:
        save_all_image_spectrum_plots(
            stats_by_model,
            output_dir,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            top_k_eig=args.top_k_eig,
            dpi=args.dpi,
            exclude_cls=args.exclude_cls,
            center_per_image=args.center_per_image,
        )
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with (metadata_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        save_image_spectrum_plot_data(
            stats_by_model,
            summary,
            metadata_dir / "plot_data.json",
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            top_k_eig=args.top_k_eig,
            exclude_cls=args.exclude_cls,
            center_per_image=args.center_per_image,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            n_images=args.n_images,
            seed=args.seed,
        )

    logger.info("Wrote image spectra to %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
