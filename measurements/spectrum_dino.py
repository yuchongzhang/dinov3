from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinov3.data.transforms import CROP_DEFAULT_SIZE, RESIZE_DEFAULT_SIZE
from measurements.dino_measurement_utils import (
    autocast_context,
    build_eval_dataset,
    infer_config_file,
    infer_model_name,
    load_config,
    load_model_for_measurement,
    make_simple_data_loader,
    safe_output_stem,
)


logger = logging.getLogger("measurements.spectrum_dino")

DEFAULT_MODEL_DIRS = (
    "vits_claude",
    "vits_1L_claude",
    "vits_3L_claude",
    "vits_5L_claude",
    "vits_6L_claude",
)

PLOT_FONT_RC = {
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

NORMALIZATION_MODES = ((False, "unnormalized"), (True, "normalized"))
PLOT_WINDOWS = ("full", "cutoff")
SUBSET_COMPARISON_LABELS = ("ViT-S", "ViT-S-5L")
SPECTRUM_LINEWIDTH = 4.0
MODEL_COLOR_PALETTE = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D")
MODEL_COLOR_ASSIGNMENTS = {
    "ViT-S": MODEL_COLOR_PALETTE[0],
    "ViT-S-5L": MODEL_COLOR_PALETTE[1],
    "ViT-S-6L": MODEL_COLOR_PALETTE[2],
    "ViT-S-1L": MODEL_COLOR_PALETTE[3],
    "ViT-S-3L": MODEL_COLOR_PALETTE[4],
}


def prepare_matplotlib_cache() -> None:
    cache_root = Path(f"/tmp/spectrum_dino_plot_cache_{os.getuid()}")
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


def safe_plot_stem(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def token_scope_description(*, exclude_cls: bool) -> str:
    return "patch tokens" if exclude_cls else "CLS+patch tokens"


def output_mode_name(*, exclude_cls: bool, center_suffix: str | None = None) -> str:
    base = "patch_tokens" if exclude_cls else "cls_plus_patch_tokens"
    if center_suffix:
        return f"{base}_{center_suffix}"
    return base


def model_color_map(names: Iterable[str]) -> dict[str, str]:
    colors = {}
    for idx, name in enumerate(names):
        canonical_name = re.sub(r"_\d+$", "", name)
        colors[name] = MODEL_COLOR_ASSIGNMENTS.get(
            canonical_name,
            MODEL_COLOR_PALETTE[idx % len(MODEL_COLOR_PALETTE)],
        )
    return colors


def save_figure_png_and_pdf(fig, output_path: Path) -> None:
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ImageNet-val eigenvalues of sum_i X_i^T X_i for DINOv3 last-layer token features."
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
    parser.add_argument("--output-dir", type=Path, default=Path("measurements") / "dino_spectrum_results")
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
        type=int,
        default=-1,
        help="Only visualize the top k eigenvalues by index. Use -1 to plot all eigenvalues.",
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
        "--center-global",
        action="store_true",
        help="Subtract the global mean token embedding before accumulating the aggregate Gram matrix.",
    )
    parser.add_argument(
        "--save-values",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated compatibility flag. JSON summaries are always saved and .pt files are never saved.",
    )
    return parser.parse_args(argv)


def checkpoint_from_model_dir(model_dir: Path, checkpoint_step: int) -> Path:
    return model_dir / "eval" / f"training_{checkpoint_step}" / "teacher_checkpoint.pth"


def resolve_checkpoint_specs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.checkpoints is None:
        return [
            (checkpoint_from_model_dir(model_dir, args.checkpoint_step), model_dir / "config.yaml")
            for model_dir in args.model_dirs
        ]

    if args.config_files is not None and len(args.config_files) != len(args.checkpoints):
        raise ValueError("--config-files must have the same length as --checkpoints.")

    specs = []
    for idx, checkpoint in enumerate(args.checkpoints):
        config_file = args.config_files[idx] if args.config_files is not None else infer_config_file(checkpoint)
        specs.append((checkpoint, config_file))
    return specs


def get_validation_dataset_string(config: Any, override: str | None) -> str:
    if override:
        return override

    dataset = OmegaConf.select(config, "train.dataset_path") or OmegaConf.select(config, "train.dataset")
    if not dataset:
        raise ValueError("No dataset was provided and neither train.dataset_path nor train.dataset exists in config.")

    dataset = str(dataset)
    val_dataset = re.sub(r"split=TRAIN\b", "split=VAL", dataset)
    if val_dataset == dataset and "split=" not in dataset:
        logger.warning("Could not infer a validation split from %r; using it as-is.", dataset)
    return val_dataset


@torch.inference_mode()
def extract_last_layer_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    autocast_dtype: torch.dtype,
    exclude_cls: bool,
) -> torch.Tensor:
    with autocast_context(images.device, autocast_dtype):
        outputs = model.get_intermediate_layers(images, n=1, return_class_token=True, norm=True)

    if len(outputs) != 1:
        raise RuntimeError(f"Expected one intermediate layer, got {len(outputs)}")

    layer_output = outputs[0]
    class_tokens = None
    patch_tokens = layer_output
    if isinstance(layer_output, (tuple, list)):
        patch_tokens = layer_output[0]
        if len(layer_output) >= 2:
            class_tokens = layer_output[1]

    if patch_tokens.ndim == 4:
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch tokens with shape (B, T, C), got {tuple(patch_tokens.shape)}")

    if exclude_cls:
        return patch_tokens.float()
    if class_tokens is None:
        raise ValueError("Expected class tokens because --exclude-cls was not set.")
    if class_tokens.ndim != 2:
        raise ValueError(f"Expected CLS tokens with shape (B, C), got {tuple(class_tokens.shape)}")
    return torch.cat((class_tokens.unsqueeze(1), patch_tokens), dim=1).float()


def gram_eigenvalues(gram_sum: torch.Tensor) -> torch.Tensor:
    gram_sum = 0.5 * (gram_sum + gram_sum.T)
    spectrum = torch.linalg.eigvalsh(gram_sum).clamp_min(0)
    spectrum, _ = torch.sort(spectrum, descending=True)
    return spectrum.float()


def spectrum_for_plot(spectrum: torch.Tensor, top_k_eig: int) -> torch.Tensor:
    if top_k_eig > 0:
        return spectrum[:top_k_eig]
    return spectrum


def cutoff_length_for_spectrum(spectrum: torch.Tensor) -> int:
    values = spectrum.detach().double().flatten()
    if values.numel() == 0:
        return 0

    positive = values[values > 0]
    if positive.numel() < 2:
        return int(values.numel())

    log_values = positive.log()
    drops = log_values[:-1] - log_values[1:]
    return int(torch.argmax(drops).item()) + 1


def _validate_or_set_token_shape(
    *,
    tokens_per_image: int | None,
    embed_dim: int | None,
    n_tokens: int,
    dim: int,
) -> tuple[int, int]:
    if tokens_per_image is None or embed_dim is None:
        return n_tokens, dim
    if n_tokens != tokens_per_image or dim != embed_dim:
        raise ValueError(f"Token shape changed from {(tokens_per_image, embed_dim)} to {(n_tokens, dim)}.")
    return tokens_per_image, embed_dim


@torch.inference_mode()
def compute_global_token_mean(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    n_images: int,
    exclude_cls: bool,
) -> tuple[torch.Tensor, int, int, int]:
    token_sum: torch.Tensor | None = None
    token_count = 0
    num_images = 0
    tokens_per_image: int | None = None
    embed_dim: int | None = None

    for batch_idx, (samples, _targets) in enumerate(data_loader):
        if n_images > 0 and num_images >= n_images:
            break
        if n_images > 0 and num_images + samples.shape[0] > n_images:
            samples = samples[: n_images - num_images]

        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("global-mean batch %d / %d, images=%d", batch_idx, len(data_loader), num_images)

        samples = samples.to(device, non_blocking=True)
        tokens = extract_last_layer_tokens(
            model,
            samples,
            autocast_dtype=autocast_dtype,
            exclude_cls=exclude_cls,
        )
        batch_size, n_tokens, dim = tokens.shape
        tokens_per_image, embed_dim = _validate_or_set_token_shape(
            tokens_per_image=tokens_per_image,
            embed_dim=embed_dim,
            n_tokens=n_tokens,
            dim=dim,
        )

        flat_tokens = tokens.reshape(-1, dim).double()
        batch_sum = flat_tokens.sum(dim=0).cpu()
        token_sum = batch_sum if token_sum is None else token_sum + batch_sum
        token_count += flat_tokens.shape[0]
        num_images += batch_size

    if num_images == 0 or token_sum is None or token_count == 0 or tokens_per_image is None or embed_dim is None:
        raise RuntimeError("No tokens were processed while computing the global mean.")

    return (token_sum / token_count).float(), num_images, tokens_per_image, embed_dim


@torch.inference_mode()
def compute_dataset_spectrum(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    n_images: int,
    exclude_cls: bool,
    global_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int, int]:
    gram_sum: torch.Tensor | None = None
    num_images = 0
    tokens_per_image: int | None = None
    embed_dim: int | None = None
    global_mean_device: torch.Tensor | None = None

    if global_mean is not None:
        if global_mean.ndim != 1:
            raise ValueError(f"Expected global_mean with shape (C,), got {tuple(global_mean.shape)}")

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

        tokens_per_image, embed_dim = _validate_or_set_token_shape(
            tokens_per_image=tokens_per_image,
            embed_dim=embed_dim,
            n_tokens=n_tokens,
            dim=dim,
        )
        if global_mean is not None:
            if global_mean.numel() != dim:
                raise ValueError(f"Expected global_mean with length {dim}, got {global_mean.numel()}.")
            if global_mean_device is None or global_mean_device.device != tokens.device:
                global_mean_device = global_mean.to(device=tokens.device, dtype=tokens.dtype)

        flat_tokens = tokens.reshape(-1, dim)
        if global_mean_device is not None:
            flat_tokens = flat_tokens - global_mean_device
        batch_gram_sum = flat_tokens.T @ flat_tokens
        batch_gram_sum = batch_gram_sum.double().cpu()
        gram_sum = batch_gram_sum if gram_sum is None else gram_sum + batch_gram_sum
        num_images += batch_size

    if num_images == 0 or tokens_per_image is None or embed_dim is None:
        raise RuntimeError("No images were processed.")

    assert gram_sum is not None
    return gram_eigenvalues(gram_sum), num_images, tokens_per_image, embed_dim


def _spectrum_values_for_plot(
    spectrum: torch.Tensor,
    *,
    normalize: bool,
    y_scale: str,
    top_k_eig: int,
    plot_window: str,
) -> torch.Tensor:
    values = spectrum_for_plot(spectrum, top_k_eig).clone()
    if plot_window == "cutoff":
        values = values[: cutoff_length_for_spectrum(values)]
    elif plot_window != "full":
        raise ValueError(f"Unsupported plot_window: {plot_window}")

    if normalize:
        values = values / values.max().clamp_min(1e-30)

    if y_scale == "log":
        positive = values[values > 0]
        floor = positive.min() * 1e-6 if positive.numel() else torch.tensor(1e-30, dtype=values.dtype)
        values = values.clamp_min(floor.clamp_min(1e-30))
    return values


def _aggregate_ylabel(*, normalize: bool, center_global: bool) -> str:
    ylabel = "Eigenvalue of $\\sum_i X_i^\\top X_i$"
    if center_global:
        ylabel = "Eigenvalue of $\\sum_i \\tilde X_i^\\top \\tilde X_i$"
    if normalize:
        ylabel = f"Normalized {ylabel[0].lower()}{ylabel[1:]}"
    return ylabel


def save_single_spectrum_figure(
    spectrum: torch.Tensor,
    model_name: str,
    output_path: Path,
    *,
    x_scale: str,
    y_scale: str,
    normalize: bool,
    top_k_eig: int,
    plot_window: str,
    dpi: int,
    center_global: bool,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = _spectrum_values_for_plot(
        spectrum,
        normalize=normalize,
        y_scale=y_scale,
        top_k_eig=top_k_eig,
        plot_window=plot_window,
    )
    indices = torch.arange(1, values.numel() + 1)

    with plt.rc_context(PLOT_FONT_RC):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        ax.plot(indices.numpy(), values.numpy(), marker="", linewidth=SPECTRUM_LINEWIDTH)
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel(_aggregate_ylabel(normalize=normalize, center_global=center_global))
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save_figure_png_and_pdf(fig, output_path)
        plt.close(fig)


def save_combined_spectrum_figure(
    spectra: dict[str, torch.Tensor],
    output_path: Path,
    *,
    x_scale: str,
    y_scale: str,
    normalize: bool,
    top_k_eig: int,
    plot_window: str,
    dpi: int,
    center_global: bool,
    title: str,
    color_by_name: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_FONT_RC):
        fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
        for name, spectrum in spectra.items():
            values = _spectrum_values_for_plot(
                spectrum,
                normalize=normalize,
                y_scale=y_scale,
                top_k_eig=top_k_eig,
                plot_window=plot_window,
            )
            indices = torch.arange(1, values.numel() + 1)
            ax.plot(
                indices.numpy(),
                values.numpy(),
                label=name,
                color=color_by_name[name],
                linewidth=SPECTRUM_LINEWIDTH,
            )
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel(_aggregate_ylabel(normalize=normalize, center_global=center_global))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=32)
        fig.tight_layout()
        save_figure_png_and_pdf(fig, output_path)
        plt.close(fig)


def save_all_spectrum_plots(
    spectra: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    x_scale: str,
    y_scale: str,
    top_k_eig: int,
    dpi: int,
    exclude_cls: bool,
    center_global: bool,
) -> None:
    color_by_name = model_color_map(spectra)
    for name, spectrum in spectra.items():
        for plot_window in PLOT_WINDOWS:
            for normalize, normalize_dir in NORMALIZATION_MODES:
                save_single_spectrum_figure(
                    spectrum,
                    name,
                    output_dir / "individual" / plot_window / normalize_dir / f"{safe_plot_stem(name)}.png",
                    x_scale=x_scale,
                    y_scale=y_scale,
                    normalize=normalize,
                    top_k_eig=top_k_eig,
                    plot_window=plot_window,
                    dpi=dpi,
                    center_global=center_global,
                )

    combined_specs = [("all_models", spectra, "ImageNet-val aggregate spectrum across models")]
    subset = {label: spectra[label] for label in SUBSET_COMPARISON_LABELS if label in spectra}
    if len(subset) == len(SUBSET_COMPARISON_LABELS):
        combined_specs.append(
            (
                "ViT-S_vs_ViT-S-5L",
                subset,
                "ImageNet-val aggregate spectrum: ViT-S vs ViT-S-5L",
            )
        )
    else:
        missing = [label for label in SUBSET_COMPARISON_LABELS if label not in spectra]
        logger.warning("Skipping ViT-S vs ViT-S-5L spectrum plot because labels are missing: %s", ", ".join(missing))

    for stem, group_spectra, title in combined_specs:
        for plot_window in PLOT_WINDOWS:
            for normalize, normalize_dir in NORMALIZATION_MODES:
                save_combined_spectrum_figure(
                    group_spectra,
                    output_dir / "combined" / plot_window / normalize_dir / f"{stem}.png",
                    x_scale=x_scale,
                    y_scale=y_scale,
                    normalize=normalize,
                    top_k_eig=top_k_eig,
                    plot_window=plot_window,
                    dpi=dpi,
                    center_global=center_global,
                    title=title,
                    color_by_name=color_by_name,
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
    top_k_eig: int,
    exclude_cls: bool,
    center_global: bool,
) -> tuple[str, torch.Tensor, dict[str, Any]]:
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
    global_mean: torch.Tensor | None = None
    global_mean_norm: float | None = None
    if center_global:
        mean_loader = make_simple_data_loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        global_mean, mean_num_images, mean_tokens_per_image, mean_embed_dim = compute_global_token_mean(
            model=model,
            data_loader=mean_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            log_every=log_every,
            n_images=n_images,
            exclude_cls=exclude_cls,
        )
        global_mean_norm = float(global_mean.norm().item())
        logger.info(
            "%s: global mean pass images=%d, tokens/image=%d, embed_dim=%d, mean_norm=%.6g",
            model_name,
            mean_num_images,
            mean_tokens_per_image,
            mean_embed_dim,
            global_mean_norm,
        )
        del mean_loader

    data_loader = make_simple_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )

    spectrum, num_images, tokens_per_image, embed_dim = compute_dataset_spectrum(
        model=model,
        data_loader=data_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        log_every=log_every,
        n_images=n_images,
        exclude_cls=exclude_cls,
        global_mean=global_mean,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_stem = short_model_label(model_name, checkpoint_path)

    logger.info(
        "%s: images=%d, requested_n_images=%d, exclude_cls=%s, center_global=%s, tokens/image=%d, "
        "embed_dim=%d, top=%.4g, bottom=%.4g",
        model_name,
        num_images,
        n_images,
        exclude_cls,
        center_global,
        tokens_per_image,
        embed_dim,
        float(spectrum[0]),
        float(spectrum[-1]),
    )
    summary = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_string,
        "num_images": num_images,
        "requested_n_images": n_images,
        "top_k_eig": top_k_eig,
        "exclude_cls": exclude_cls,
        "center_global": center_global,
        "global_mean_norm": global_mean_norm,
        "tokens_per_image": tokens_per_image,
        "embed_dim": embed_dim,
        "rank": int(spectrum.numel()),
        "plotted_eigenvalues": int(spectrum_for_plot(spectrum, top_k_eig).numel()),
        "cutoff_eigenvalues": cutoff_length_for_spectrum(spectrum_for_plot(spectrum, top_k_eig)),
        "top": float(spectrum[0]),
        "bottom": float(spectrum[-1]),
    }
    return output_stem, spectrum, summary


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

    center_suffix = "global_centered" if args.center_global else None
    output_dir = args.output_dir / output_mode_name(exclude_cls=args.exclude_cls, center_suffix=center_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    spectra: dict[str, torch.Tensor] = {}
    summary: dict[str, dict[str, Any]] = {}
    for checkpoint_path, config_file in resolve_checkpoint_specs(args):
        name, spectrum, model_summary = run_for_checkpoint(
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
            center_global=args.center_global,
        )
        name = unique_label(name, set(spectra))
        model_summary["display_name"] = name
        spectra[name] = spectrum
        summary[name] = model_summary

    if spectra:
        save_all_spectrum_plots(
            spectra,
            output_dir,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            top_k_eig=args.top_k_eig,
            dpi=args.dpi,
            exclude_cls=args.exclude_cls,
            center_global=args.center_global,
        )
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with (metadata_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    logger.info("Wrote spectra to %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
