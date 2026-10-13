from __future__ import annotations

import argparse
import colorsys
import json
import logging
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

from dinov3.data import make_dataset
from measurements.dino_measurement_utils import (
    autocast_context,
    infer_config_file,
    infer_model_name,
    load_config,
    safe_output_stem,
)
from measurements.pca_dino_3d import (
    DEFAULT_MODEL_DIRS,
    ImageNetSample,
    PC_OVERLAY_COLORS,
    checkpoint_from_model_dir,
    get_validation_dataset_string,
    load_imagenet_sample,
    load_model_for_pca3d,
    sample_label,
    save_original_image,
    save_patch_colored_image,
)


logger = logging.getLogger("measurements.laplacian_dino")

DEFAULT_TEMPERATURE = 0.20
DEFAULT_N_EIGENVECTORS = 8
MIN_RGB_EIGENVECTORS = 4
EIGENVECTOR_BASE_COLORS = (
    ("white", (1.0, 1.0, 1.0)),
    ("red", (1.0, 0.0, 0.0)),
    ("green", (0.0, 1.0, 0.0)),
    ("blue", (0.0, 0.0, 1.0)),
    ("grey", (0.5, 0.5, 0.5)),
    ("yellow", (1.0, 0.9, 0.0)),
    ("cyan", (0.0, 0.9, 1.0)),
    ("magenta", (1.0, 0.0, 0.9)),
    ("orange", (1.0, 0.55, 0.0)),
    ("purple", (0.55, 0.25, 1.0)),
    ("teal", (0.0, 0.75, 0.65)),
    ("pink", (1.0, 0.45, 0.75)),
)


@dataclass
class LaplacianSpectrum:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    all_eigenvalues: torch.Tensor
    log_degrees: torch.Tensor


@dataclass
class LaplacianRunResult:
    model_name: str
    output_stem: str
    eigenvalues: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one ImageNet-val image using the smallest random-walk Laplacian "
            "eigenvectors of DINOv3 token affinity graphs."
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
            "Dataset string. Defaults to train.dataset_path/train.dataset from the first config with "
            "split=TRAIN replaced by split=VAL."
        ),
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="ImageNet-val dataset index to visualize. Negative indices count from the end.",
    )
    parser.add_argument(
        "--random-image",
        action="store_true",
        help="Sample a validation image index uniformly with --seed instead of using --image-index.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used with --random-image.")
    parser.add_argument("--output-dir", type=Path, default=Path("measurements") / "dino_laplacian_results")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--include-cls",
        action="store_true",
        help="Include the leading CLS token in the graph eigensolve. Spatial heatmaps still render patches only.",
    )
    parser.add_argument(
        "--n-eigenvectors",
        type=int,
        default=DEFAULT_N_EIGENVECTORS,
        help="Number of smallest Laplacian eigenvectors to compute and visualize individually.",
    )
    parser.add_argument(
        "--patch-color-alpha",
        type=float,
        default=0.65,
        help="Opacity of the RGB spectral patch colors over the resized original image.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def prepare_matplotlib_cache() -> None:
    cache_root = Path(f"/tmp/laplacian_dino_plot_cache_{os.getuid()}")
    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib")))
    xdg_cache_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg")))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)


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


def safe_float_for_path(value: float) -> str:
    text = f"{value:.6g}"
    text = text.replace("-", "m").replace(".", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def image_output_dir(output_dir: Path, sample: ImageNetSample) -> Path:
    return output_dir / f"imagenet_val_{sample.index:05d}"


def spectral_output_dir(output_dir: Path, sample: ImageNetSample, *, temperature: float, include_cls: bool) -> Path:
    cls_dirname = "with_cls" if include_cls else "no_cls"
    return image_output_dir(output_dir, sample) / f"temp_{safe_float_for_path(temperature)}" / cls_dirname


def l2_normalize_token_embeddings(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"Expected token embeddings with shape (T, D), got {tuple(tokens.shape)}.")
    if tokens.shape[0] == 0:
        raise ValueError("Expected at least one token embedding.")
    if tokens.shape[1] == 0:
        raise ValueError("Expected token embeddings with nonzero feature dimension.")
    return torch.nn.functional.normalize(tokens.detach().to(dtype=torch.float64), p=2, dim=1, eps=1e-12)


def pairwise_cosine_logits(normalized_tokens: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"--temperature must be positive, got {temperature}.")
    if normalized_tokens.ndim != 2:
        raise ValueError(f"Expected normalized tokens with shape (T, D), got {tuple(normalized_tokens.shape)}.")
    logits = normalized_tokens @ normalized_tokens.T
    return logits / float(temperature)


def random_walk_transition_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"Expected square logits matrix, got {tuple(logits.shape)}.")
    return torch.softmax(logits, dim=1)


def random_walk_laplacian_from_logits(logits: torch.Tensor) -> torch.Tensor:
    transition = random_walk_transition_from_logits(logits)
    identity = torch.eye(logits.shape[0], dtype=transition.dtype, device=transition.device)
    return identity - transition


def symmetric_normalized_affinity_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"Expected square logits matrix, got {tuple(logits.shape)}.")
    log_degrees = torch.logsumexp(logits, dim=1)
    log_affinity = logits - 0.5 * (log_degrees[:, None] + log_degrees[None, :])
    affinity = torch.exp(log_affinity)
    affinity = 0.5 * (affinity + affinity.T)
    return affinity, log_degrees


def smallest_symmetric_eigenpairs(matrix: torch.Tensor, *, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {tuple(matrix.shape)}.")
    n = matrix.shape[0]
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}.")

    try:
        import scipy.linalg

        values_np, vectors_np = scipy.linalg.eigh(
            matrix.detach().cpu().numpy(),
            subset_by_index=(0, k - 1),
            driver="evr",
        )
        values = torch.from_numpy(values_np).to(dtype=matrix.dtype)
        vectors = torch.from_numpy(vectors_np).to(dtype=matrix.dtype)
        values = values.to(device=matrix.device)
        vectors = vectors.to(device=matrix.device)
        return values, vectors
    except Exception as exc:
        logger.debug("Falling back to torch.linalg.eigh for k=%d: %s", k, exc)
        values, vectors = torch.linalg.eigh(matrix)
        return values[:k], vectors[:, :k]


def all_symmetric_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {tuple(matrix.shape)}.")

    try:
        import scipy.linalg

        values_np = scipy.linalg.eigvalsh(matrix.detach().cpu().numpy())
        values = torch.from_numpy(values_np).to(dtype=matrix.dtype, device=matrix.device)
        return values
    except Exception as exc:
        logger.debug("Falling back to torch.linalg.eigvalsh for full spectrum: %s", exc)
        return torch.linalg.eigvalsh(matrix)


def orient_eigenvectors(eigenvectors: torch.Tensor) -> torch.Tensor:
    oriented = eigenvectors.clone()
    for column_idx in range(oriented.shape[1]):
        column = oriented[:, column_idx]
        pivot_idx = int(torch.argmax(column.abs()).item())
        if column[pivot_idx] < 0:
            oriented[:, column_idx] = -column
    return oriented


def compute_laplacian_spectrum(
    token_features: torch.Tensor,
    *,
    temperature: float,
    n_eigenvectors: int,
) -> LaplacianSpectrum:
    if n_eigenvectors < 1:
        raise ValueError(f"--n-eigenvectors must be positive, got {n_eigenvectors}.")

    normalized_tokens = l2_normalize_token_embeddings(token_features)
    n_tokens = normalized_tokens.shape[0]
    if n_eigenvectors > n_tokens:
        raise ValueError(
            f"Requested {n_eigenvectors} eigenvectors, but the graph has only {n_tokens} tokens."
        )

    logits = pairwise_cosine_logits(normalized_tokens, temperature=temperature)
    sym_affinity, log_degrees = symmetric_normalized_affinity_from_logits(logits)
    identity = torch.eye(n_tokens, dtype=sym_affinity.dtype, device=sym_affinity.device)
    symmetric_laplacian = identity - sym_affinity
    symmetric_laplacian = 0.5 * (symmetric_laplacian + symmetric_laplacian.T)

    eigenvalues, symmetric_eigenvectors = smallest_symmetric_eigenpairs(
        symmetric_laplacian,
        k=n_eigenvectors,
    )
    if n_eigenvectors == n_tokens:
        all_eigenvalues = eigenvalues
    else:
        all_eigenvalues = all_symmetric_eigenvalues(symmetric_laplacian)
    random_walk_eigenvectors = symmetric_eigenvectors * torch.exp(-0.5 * log_degrees).unsqueeze(1)
    random_walk_eigenvectors = orient_eigenvectors(random_walk_eigenvectors)

    return LaplacianSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=random_walk_eigenvectors,
        all_eigenvalues=all_eigenvalues,
        log_degrees=log_degrees,
    )


def patch_eigenvectors_for_render(
    eigenvectors: torch.Tensor,
    *,
    include_cls: bool,
    expected_patches: int,
) -> torch.Tensor:
    patch_eigenvectors = eigenvectors[1:] if include_cls else eigenvectors
    if patch_eigenvectors.shape[0] != expected_patches:
        raise ValueError(
            f"Patch eigenvectors have {patch_eigenvectors.shape[0]} rows, expected {expected_patches}."
        )
    return patch_eigenvectors


def heatmap_values(values: torch.Tensor, *, force_full_intensity: bool = False) -> torch.Tensor:
    values = values.detach().to(dtype=torch.float32)
    if force_full_intensity:
        return torch.ones_like(values)

    min_value = values.min()
    max_value = values.max()
    value_range = max_value - min_value
    if value_range <= 1e-12:
        return torch.ones_like(values)
    return (values - min_value) / value_range


def eigenvector_overlay_color(eigenvector_index: int) -> tuple[str, torch.Tensor]:
    if eigenvector_index < len(EIGENVECTOR_BASE_COLORS):
        color_name, color = EIGENVECTOR_BASE_COLORS[eigenvector_index]
    elif eigenvector_index < len(PC_OVERLAY_COLORS):
        color_name, color = PC_OVERLAY_COLORS[eigenvector_index]
    else:
        hue = (0.08 + (eigenvector_index - len(EIGENVECTOR_BASE_COLORS)) * 0.61803398875) % 1.0
        color_name = f"hue_{int(round(hue * 360.0)):03d}"
        color = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return color_name, torch.tensor(color, dtype=torch.float32)


def eigenvector_patch_rgb(patch_eigenvectors: torch.Tensor, eigenvector_index: int) -> tuple[str, torch.Tensor]:
    if eigenvector_index < 0 or eigenvector_index >= patch_eigenvectors.shape[1]:
        raise ValueError(
            f"Eigenvector index {eigenvector_index} is outside range [0, {patch_eigenvectors.shape[1] - 1}]."
        )
    color_name, color = eigenvector_overlay_color(eigenvector_index)
    values = heatmap_values(
        patch_eigenvectors[:, eigenvector_index],
        force_full_intensity=(eigenvector_index == 0),
    ).unsqueeze(1)
    return color_name, values.mul(color.unsqueeze(0))


def rgb_overlay_from_eigenvectors(patch_eigenvectors: torch.Tensor) -> torch.Tensor:
    if patch_eigenvectors.ndim != 2:
        raise ValueError(f"Expected patch eigenvectors with shape (T, K), got {tuple(patch_eigenvectors.shape)}.")
    if patch_eigenvectors.shape[1] < MIN_RGB_EIGENVECTORS:
        raise ValueError(
            f"Need at least {MIN_RGB_EIGENVECTORS} eigenvectors to render RGB from u2, u3, u4."
        )
    channels = [
        heatmap_values(patch_eigenvectors[:, eigenvector_index])
        for eigenvector_index in (1, 2, 3)
    ]
    return torch.stack(channels, dim=1)


@torch.inference_mode()
def extract_final_tokens(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
    include_cls: bool,
) -> torch.Tensor:
    image_batch = image.unsqueeze(0).to(device, non_blocking=True)
    with autocast_context(device, autocast_dtype):
        outputs = model.get_intermediate_layers(
            image_batch,
            n=1,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

    if len(outputs) != 1:
        raise RuntimeError(f"Expected one final-layer output, got {len(outputs)}.")

    layer_output = outputs[0]
    class_token = None
    patch_tokens = layer_output
    if isinstance(layer_output, (tuple, list)):
        patch_tokens = layer_output[0]
        if len(layer_output) >= 2:
            class_token = layer_output[1]

    if patch_tokens.ndim == 4:
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
    if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
        raise ValueError(f"Expected patch tokens with shape (1, T, C), got {tuple(patch_tokens.shape)}.")

    tokens = patch_tokens.squeeze(0)
    if include_cls:
        if class_token is None:
            raise ValueError("Expected a CLS token because --include-cls was set.")
        if class_token.ndim != 2 or class_token.shape[0] != 1:
            raise ValueError(f"Expected CLS token with shape (1, C), got {tuple(class_token.shape)}.")
        tokens = torch.cat((class_token.squeeze(0).unsqueeze(0), tokens), dim=0)
    return tokens.detach().cpu().float()


def save_eigenvalue_spectrum(
    eigenvalues: torch.Tensor,
    output_path: Path,
    *,
    model_name: str,
    sample: ImageNetSample,
    temperature: float,
    include_cls: bool,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    values = eigenvalues.detach().cpu().float()
    indices = torch.arange(1, values.numel() + 1)
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=dpi)
    ax.plot(indices.numpy(), values.numpy(), marker="o", linewidth=1.5, markersize=3.5)
    ax.set_xlabel("Eigenvector index")
    ax.set_ylabel("Laplacian eigenvalue")
    ax.set_title(
        f"{model_name}\n{sample_label(sample)}\n"
        f"temperature={temperature:g}, {'with CLS' if include_cls else 'patches only'}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_combined_eigenvalue_spectrum(
    run_results: list[LaplacianRunResult],
    output_path: Path,
    *,
    sample: ImageNetSample,
    temperature: float,
    include_cls: bool,
    dpi: int,
) -> None:
    if not run_results:
        raise ValueError("Cannot save a combined spectrum figure without any model results.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=dpi)
    cmap = plt.get_cmap("viridis")
    n_models = len(run_results)
    for idx, result in enumerate(run_results):
        values = result.eigenvalues.detach().cpu().float()
        indices = torch.arange(1, values.numel() + 1)
        ax.plot(
            indices.numpy(),
            values.numpy(),
            marker="o",
            linewidth=1.4,
            markersize=3.0,
            label=result.output_stem,
            color=cmap(idx / max(n_models - 1, 1)),
        )

    ax.set_xlabel("Eigenvector index")
    ax.set_ylabel("Laplacian eigenvalue")
    ax.set_title(
        f"Full Laplacian eigenvalue spectra across models\n{sample_label(sample)}\n"
        f"temperature={temperature:g}, {'with CLS' if include_cls else 'patches only'}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def metadata_payload(
    *,
    model_name: str,
    output_stem: str,
    checkpoint_path: Path,
    config_file: Path,
    sample: ImageNetSample,
    temperature: float,
    include_cls: bool,
    n_eigenvectors: int,
    token_count: int,
    spectrum: LaplacianSpectrum,
    outputs: dict[str, str],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "output_stem": output_stem,
        "checkpoint": str(checkpoint_path),
        "config_file": str(config_file),
        "image_index": sample.index,
        "target": sample.target,
        "class_id": sample.class_id,
        "class_name": sample.class_name,
        "patch_grid": list(sample.patch_grid),
        "temperature": temperature,
        "include_cls": include_cls,
        "graph_token_count": token_count,
        "patch_token_count": sample.patch_grid[0] * sample.patch_grid[1],
        "n_eigenvectors": n_eigenvectors,
        "visualized_eigenvalues": [float(value) for value in spectrum.eigenvalues.detach().cpu().tolist()],
        "spectrum_eigenvalues": [float(value) for value in spectrum.all_eigenvalues.detach().cpu().tolist()],
        "outputs": outputs,
        "math": {
            "token_preprocessing": "l2_normalized_last_layer_tokens",
            "adjacency": "W_ij = exp(x_i^T x_j / temperature), including self-loops",
            "random_walk_transition": "D^{-1}W = softmax((X X^T) / temperature, dim=1)",
            "eigensolve": "computed via symmetric normalized Laplacian and converted to random-walk eigenvectors",
        },
    }


def write_metadata(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_for_checkpoint(
    *,
    checkpoint_path: Path,
    config_file: Path,
    sample: ImageNetSample,
    output_dir: Path,
    device: torch.device,
    temperature: float,
    include_cls: bool,
    n_eigenvectors: int,
    patch_color_alpha: float,
    dpi: int,
) -> LaplacianRunResult:
    checkpoint_path = checkpoint_path.resolve()
    config_file = config_file.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing config file: {config_file}")

    config = load_config(config_file)
    model_name = infer_model_name(config, config_file)
    logger.info("Loading %s from %s", model_name, checkpoint_path)
    model, autocast_dtype = load_model_for_pca3d(config, checkpoint_path, device)

    token_features = extract_final_tokens(
        model,
        sample.image,
        device=device,
        autocast_dtype=autocast_dtype,
        include_cls=include_cls,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    expected_patches = sample.patch_grid[0] * sample.patch_grid[1]
    graph_token_count = expected_patches + (1 if include_cls else 0)
    if token_features.shape[0] != graph_token_count:
        raise ValueError(
            f"Extracted {token_features.shape[0]} graph tokens, expected {graph_token_count} "
            f"for patch grid {sample.patch_grid} and include_cls={include_cls}."
        )

    spectrum = compute_laplacian_spectrum(
        token_features,
        temperature=temperature,
        n_eigenvectors=n_eigenvectors,
    )
    patch_eigenvectors = patch_eigenvectors_for_render(
        spectrum.eigenvectors,
        include_cls=include_cls,
        expected_patches=expected_patches,
    )

    output_stem = safe_output_stem(model_name, checkpoint_path)
    patch_size_pixels = sample.image_resized.shape[-2] // sample.patch_grid[0]
    outputs: dict[str, str] = {}

    rgb_path = output_dir / "rgb_u02_u03_u04" / f"{output_stem}.png"
    save_patch_colored_image(
        rgb_overlay_from_eigenvectors(patch_eigenvectors),
        sample,
        rgb_path,
        patch_size=patch_size_pixels,
        alpha=patch_color_alpha,
    )
    outputs["rgb_u02_u03_u04"] = str(rgb_path)

    rgb_full_path = output_dir / "rgb_u02_u03_u04_alpha_1_0" / f"{output_stem}.png"
    save_patch_colored_image(
        rgb_overlay_from_eigenvectors(patch_eigenvectors),
        sample,
        rgb_full_path,
        patch_size=patch_size_pixels,
        alpha=1.0,
    )
    outputs["rgb_u02_u03_u04_alpha_1_0"] = str(rgb_full_path)

    for eigenvector_index in range(n_eigenvectors):
        color_name, patch_rgb = eigenvector_patch_rgb(patch_eigenvectors, eigenvector_index)
        eigenvector_dirname = f"u{eigenvector_index + 1:02d}_{color_name}"
        eigenvector_path = output_dir / "eigenvector_heatmaps" / eigenvector_dirname / f"{output_stem}.png"
        save_patch_colored_image(
            patch_rgb,
            sample,
            eigenvector_path,
            patch_size=patch_size_pixels,
            alpha=1.0,
        )
        outputs[eigenvector_dirname] = str(eigenvector_path)

    spectrum_path = output_dir / "eigenvalue_spectrum" / f"{output_stem}.png"
    save_eigenvalue_spectrum(
        spectrum.all_eigenvalues,
        spectrum_path,
        model_name=model_name,
        sample=sample,
        temperature=temperature,
        include_cls=include_cls,
        dpi=dpi,
    )
    outputs["eigenvalue_spectrum"] = str(spectrum_path)

    metadata_path = output_dir / "metadata" / f"{output_stem}.json"
    outputs["metadata"] = str(metadata_path)
    write_metadata(
        metadata_payload(
            model_name=model_name,
            output_stem=output_stem,
            checkpoint_path=checkpoint_path,
            config_file=config_file,
            sample=sample,
            temperature=temperature,
            include_cls=include_cls,
            n_eigenvectors=n_eigenvectors,
            token_count=token_features.shape[0],
            spectrum=spectrum,
            outputs=outputs,
        ),
        metadata_path,
    )

    for output_path in outputs.values():
        logger.info("Wrote %s", Path(output_path).resolve())

    return LaplacianRunResult(
        model_name=model_name,
        output_stem=output_stem,
        eigenvalues=spectrum.all_eigenvalues.detach().cpu().float(),
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_matplotlib_cache()

    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.n_eigenvectors < MIN_RGB_EIGENVECTORS:
        raise ValueError(f"--n-eigenvectors must be at least {MIN_RGB_EIGENVECTORS}.")
    if args.patch_color_alpha < 0 or args.patch_color_alpha > 1:
        raise ValueError("--patch-color-alpha must be in the interval [0, 1].")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if args.image_size % args.patch_size != 0:
        raise ValueError("--image-size must be divisible by --patch-size.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    checkpoint_specs = resolve_checkpoint_specs(args)
    if not checkpoint_specs:
        raise ValueError("No checkpoints were selected.")

    first_config_file = checkpoint_specs[0][1].resolve()
    if not first_config_file.is_file():
        raise FileNotFoundError(f"Missing config file: {first_config_file}")
    first_config = load_config(first_config_file)
    dataset_string = get_validation_dataset_string(first_config, args.dataset)
    logger.info("dataset=%s", dataset_string)

    dataset = make_dataset(dataset_str=dataset_string)
    sample = load_imagenet_sample(
        dataset,
        image_index=args.image_index,
        random_image=args.random_image,
        seed=args.seed,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    logger.info("selected ImageNet-val sample: %s", sample_label(sample))

    base_image_dir = image_output_dir(args.output_dir, sample)
    input_path = save_original_image(sample, base_image_dir)
    logger.info("Wrote %s", input_path.resolve())

    output_dir = spectral_output_dir(
        args.output_dir,
        sample,
        temperature=args.temperature,
        include_cls=args.include_cls,
    )
    device = torch.device(args.device)
    run_results: list[LaplacianRunResult] = []
    for checkpoint_path, config_file in checkpoint_specs:
        run_results.append(
            run_for_checkpoint(
                checkpoint_path=checkpoint_path,
                config_file=config_file,
                sample=sample,
                output_dir=output_dir,
                device=device,
                temperature=args.temperature,
                include_cls=args.include_cls,
                n_eigenvectors=args.n_eigenvectors,
                patch_color_alpha=args.patch_color_alpha,
                dpi=args.dpi,
            )
        )

    combined_spectrum_path = output_dir / "eigenvalue_spectrum" / "all_models.png"
    save_combined_eigenvalue_spectrum(
        run_results,
        combined_spectrum_path,
        sample=sample,
        temperature=args.temperature,
        include_cls=args.include_cls,
        dpi=args.dpi,
    )
    logger.info("Wrote %s", combined_spectrum_path.resolve())

    logger.info("Wrote Laplacian visualizations to %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
