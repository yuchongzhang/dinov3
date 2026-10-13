from __future__ import annotations

import argparse
import colorsys
import io
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinov3.checkpointer import init_model_from_checkpoint_for_evals
from dinov3.data import make_dataset
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_from_cfg
from measurements.dino_measurement_utils import (
    autocast_context,
    infer_config_file,
    infer_model_name,
    load_config,
    load_model_for_measurement,
    safe_output_stem,
)


logger = logging.getLogger("measurements.pca_dino_3d")

DEFAULT_MODEL_DIRS = (
    "vits_claude",
    "vits_1L_claude",
    "vits_3L_claude",
    "vits_5L_claude",
    "vits_6L_claude",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSPARENT_PATCH_COLOR_ALPHA = 0.4
PC_OVERLAY_COLORS = (
    ("red", (1.0, 0.0, 0.0)),
    ("green", (0.0, 1.0, 0.0)),
    ("blue", (0.0, 0.0, 1.0)),
    ("white", (1.0, 1.0, 1.0)),
    ("grey", (0.5, 0.5, 0.5)),
    ("yellow", (1.0, 0.9, 0.0)),
    ("cyan", (0.0, 0.9, 1.0)),
    ("magenta", (1.0, 0.0, 0.9)),
    ("orange", (1.0, 0.55, 0.0)),
    ("purple", (0.55, 0.25, 1.0)),
    ("teal", (0.0, 0.75, 0.65)),
    ("pink", (1.0, 0.45, 0.75)),
)
PC_ALPHA_IMAGES = (
    ("patch_rgba_original_rgb_pc1_alpha", 0),
    ("patch_rgba_original_rgb_pc2_alpha", 1),
    ("patch_rgba_original_rgb_pc3_alpha", 2),
)
RGB_TRIPLET_OVERLAYS = (
    ("pc04_pc05_pc06_alpha_1_0", (3, 4, 5)),
    ("pc07_pc08_pc09_alpha_1_0", (6, 7, 8)),
)
MIN_RGB_TRIPLET_COMPONENTS = (
    max(max(component_indices) for _, component_indices in RGB_TRIPLET_OVERLAYS) + 1
)


@dataclass
class ImageNetSample:
    image: torch.Tensor
    image_resized: torch.Tensor
    index: int
    target: int | None
    class_id: str | None
    class_name: str | None
    raw_image: Image.Image | None
    patch_grid: tuple[int, int]


@dataclass
class PCA3DProjection:
    coordinates: torch.Tensor
    rgb: torch.Tensor
    component_values: torch.Tensor
    explained_variance_ratio: torch.Tensor
    singular_values: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one ImageNet-val image as DINO patch-feature projections onto the top-3 PCA components. "
            "Each patch is drawn as a 3D dot whose RGB color is encoded from the same three PCA coordinates."
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
    parser.add_argument("--output-dir", type=Path, default=Path("measurements") / "dino_pca_3d_results")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument(
        "--n-eigen-vec",
        type=int,
        default=3,
        help=(
            "Number of single-eigenvector patch overlays to save. The PCA fit always includes at least 9 "
            "components for RGB triplet overlays."
        ),
    )
    parser.add_argument(
        "--point-alpha",
        "--alpha",
        dest="point_alpha",
        type=float,
        default=0.65,
        help="Opacity of the points in the interactive Plotly 3D scatter.",
    )
    parser.add_argument(
        "--patch-color-alpha",
        type=float,
        default=0.65,
        help="Opacity of the PCA RGB patch colors over the resized original image.",
    )
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--plot-width", type=int, default=900)
    parser.add_argument("--plot-height", type=int, default=760)
    parser.add_argument(
        "--plotly-js",
        choices=("inline", "cdn"),
        default="inline",
        help="How to include Plotly JavaScript in the HTML output. 'inline' is self-contained.",
    )
    return parser.parse_args()


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


def normalize_index(index: int, dataset_length: int) -> int:
    if dataset_length <= 0:
        raise RuntimeError("Validation dataset is empty.")
    if index < 0:
        index += dataset_length
    if index < 0 or index >= dataset_length:
        raise IndexError(f"--image-index {index} is outside the dataset range [0, {dataset_length - 1}].")
    return index


def maybe_int_target(target) -> int | None:
    if target is None:
        return None
    if torch.is_tensor(target):
        if target.numel() == 1:
            return int(target.item())
        return None
    try:
        return int(target)
    except (TypeError, ValueError):
        return None


def maybe_get_dataset_value(dataset, method_name: str, index: int) -> str | None:
    method = getattr(dataset, method_name, None)
    if method is None:
        return None
    try:
        value = method(index)
    except Exception as exc:
        logger.debug("Could not read %s for index %d: %s", method_name, index, exc)
        return None
    return None if value is None else str(value)


def load_raw_image(dataset, index: int) -> Image.Image | None:
    get_image_data = getattr(dataset, "get_image_data", None)
    if get_image_data is None:
        return None
    try:
        image_data = get_image_data(index)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as exc:
        logger.debug("Could not load raw image for index %d: %s", index, exc)
        return None


def resize_transform(image: Image.Image, *, image_size: int, patch_size: int) -> torch.Tensor:
    width, height = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((width * image_size) / (height * patch_size))
    if h_patches <= 0 or w_patches <= 0:
        raise ValueError(
            f"Invalid resized patch grid ({h_patches}, {w_patches}) for image size {image.size}, "
            f"--image-size {image_size}, and --patch-size {patch_size}."
        )
    return TF.to_tensor(TF.resize(image, (h_patches * patch_size, w_patches * patch_size)))


def load_imagenet_sample(
    dataset,
    *,
    image_index: int,
    random_image: bool,
    seed: int,
    image_size: int,
    patch_size: int,
) -> ImageNetSample:
    dataset_length = len(dataset)
    if random_image:
        rng = np.random.default_rng(seed)
        image_index = int(rng.integers(0, dataset_length))
    image_index = normalize_index(image_index, dataset_length)

    raw_image, target = dataset[image_index]
    if isinstance(raw_image, Image.Image):
        raw_image = raw_image.convert("RGB")
    else:
        raw_image = load_raw_image(dataset, image_index)
    if raw_image is None:
        raise RuntimeError(f"Could not load raw image data for validation sample {image_index}.")

    image_resized = resize_transform(raw_image, image_size=image_size, patch_size=patch_size)
    image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    h_patches, w_patches = [int(d / patch_size) for d in image_resized_norm.shape[1:]]

    return ImageNetSample(
        image=image_resized_norm,
        image_resized=image_resized,
        index=image_index,
        target=maybe_int_target(target),
        class_id=maybe_get_dataset_value(dataset, "get_class_id", image_index),
        class_name=maybe_get_dataset_value(dataset, "get_class_name", image_index),
        raw_image=raw_image,
        patch_grid=(h_patches, w_patches),
    )


@torch.inference_mode()
def extract_final_patch_features(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    image_batch = image.unsqueeze(0).to(device, non_blocking=True)
    with autocast_context(device, autocast_dtype):
        outputs = model.get_intermediate_layers(image_batch, n=1, reshape=False, norm=True)

    if len(outputs) != 1:
        raise RuntimeError(f"Expected one final-layer output, got {len(outputs)}.")

    patch_tokens = outputs[0]
    if isinstance(patch_tokens, (tuple, list)):
        patch_tokens = patch_tokens[0]
    if patch_tokens.ndim == 4:
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
    if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
        raise ValueError(f"Expected patch tokens with shape (1, T, C), got {tuple(patch_tokens.shape)}.")

    return patch_tokens.squeeze(0).detach().cpu().float()


def compute_pca_3d_projection(patch_features: torch.Tensor, *, n_components: int) -> PCA3DProjection:
    if patch_features.ndim != 2:
        raise ValueError(f"Expected patch features with shape (T, C), got {tuple(patch_features.shape)}.")
    if patch_features.shape[0] < 3:
        raise RuntimeError("Need at least three patches to fit a 3D PCA projection.")
    if patch_features.shape[1] < 3:
        raise RuntimeError("Need feature dimension of at least three to fit a 3D PCA projection.")
    if n_components < 3:
        raise ValueError("--n-eigen-vec must be at least 3 for the 3D PCA visualization.")
    max_components = min(patch_features.shape)
    if n_components > max_components:
        raise ValueError(
            f"Requested {n_components} PCA components, but this image/model supports at most {max_components} "
            f"(min(num_patches={patch_features.shape[0]}, feature_dim={patch_features.shape[1]}))."
        )

    pca = PCA(n_components=n_components, whiten=False)
    coordinates_np = pca.fit_transform(patch_features.numpy())
    coordinates = torch.from_numpy(coordinates_np).float()

    # Same encoding family as measurements/pca_dino.py: whitened PC coordinates are
    # squashed to valid RGB values while preserving their ordering and sign.
    component_values = torch.sigmoid(coordinates.mul(2.0)).clamp(0.0, 1.0)
    rgb = component_values[:, :3]

    return PCA3DProjection(
        coordinates=coordinates,
        rgb=rgb,
        component_values=component_values,
        explained_variance_ratio=torch.from_numpy(pca.explained_variance_ratio_).float(),
        singular_values=torch.from_numpy(pca.singular_values_).float(),
    )


def load_model_for_pca3d(
    config: Any,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.dtype]:
    if device.type == "cuda":
        return load_model_for_measurement(config, checkpoint_path, device)

    if checkpoint_path.is_dir():
        raise RuntimeError(
            "CPU loading is only implemented for consolidated teacher_checkpoint.pth files. "
            f"{checkpoint_path} is a checkpoint directory."
        )

    model, _ = build_model_from_cfg(config, only_teacher=True)
    if checkpoint_path:
        logger.info("PyTorch consolidated checkpoint")
        model.to_empty(device=device)
        init_model_from_checkpoint_for_evals(model, checkpoint_path, "teacher")
    else:
        logger.info("No pretrained weights")
        model.to_empty(device=device)
        model.init_weights()
    model.eval()
    return model, get_autocast_dtype(config)


def sample_label(sample: ImageNetSample) -> str:
    parts = [f"index {sample.index}", f"patches {sample.patch_grid[0]}x{sample.patch_grid[1]}"]
    if sample.target is not None:
        parts.append(f"target {sample.target}")
    if sample.class_id:
        parts.append(sample.class_id)
    if sample.class_name:
        parts.append(sample.class_name)
    return ", ".join(parts)


def camera_eye_from_angles(elev: float, azim: float, *, radius: float = 1.75) -> dict[str, float]:
    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)
    xy = radius * math.cos(elev_rad)
    return {
        "x": xy * math.cos(azim_rad),
        "y": xy * math.sin(azim_rad),
        "z": radius * math.sin(elev_rad),
    }


def rgb_strings(rgb: torch.Tensor) -> list[str]:
    rgb_uint8 = rgb.mul(255.0).round().clamp(0, 255).to(torch.uint8).tolist()
    return [f"rgb({r}, {g}, {b})" for r, g, b in rgb_uint8]


def patch_hover_text(sample: ImageNetSample, coordinates: torch.Tensor) -> list[str]:
    h_patches, w_patches = sample.patch_grid
    points = coordinates[:, :3].tolist()
    hover_text = []
    for idx, (pc1, pc2, pc3) in enumerate(points):
        row = idx // w_patches
        col = idx % w_patches
        hover_text.append(
            f"patch {idx}<br>row {row}, col {col}<br>"
            f"PC1 {pc1:.4f}<br>PC2 {pc2:.4f}<br>PC3 {pc3:.4f}"
        )
    if len(hover_text) != h_patches * w_patches:
        raise ValueError(
            f"Hover text contains {len(hover_text)} entries, expected {h_patches * w_patches} patches."
        )
    return hover_text


def save_pca_3d_html(
    projection: PCA3DProjection,
    output_path: Path,
    *,
    model_name: str,
    sample: ImageNetSample,
    point_size: float,
    alpha: float,
    elev: float,
    azim: float,
    plot_width: int,
    plot_height: int,
    plotly_js: str,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for rotatable 3D PCA visualizations. Install it with `pip install plotly` "
            "or reinstall the project requirements."
        ) from exc

    points = projection.coordinates.numpy()
    explained = projection.explained_variance_ratio.mul(100.0).tolist()
    title = (
        f"{model_name}<br>{sample_label(sample)}<br>"
        f"explained variance: {explained[0]:.1f}%, {explained[1]:.1f}%, {explained[2]:.1f}%"
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": point_size,
                    "color": rgb_strings(projection.rgb),
                    "opacity": alpha,
                    "line": {"width": 0},
                },
                text=patch_hover_text(sample, projection.coordinates),
                hoverinfo="text",
                name="patches",
            )
        ]
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        width=plot_width,
        height=plot_height,
        margin={"l": 0, "r": 0, "t": 92, "b": 0},
        scene={
            "xaxis": {"title": "PC1", "showspikes": False},
            "yaxis": {"title": "PC2", "showspikes": False},
            "zaxis": {"title": "PC3", "showspikes": False},
            "aspectmode": "cube",
            "camera": {"eye": camera_eye_from_angles(elev, azim)},
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_plotlyjs: bool | str = True if plotly_js == "inline" else "cdn"
    fig.write_html(output_path, include_plotlyjs=include_plotlyjs, full_html=True)


def save_patch_colored_image(
    patch_rgb: torch.Tensor,
    sample: ImageNetSample,
    output_path: Path,
    *,
    patch_size: int,
    alpha: float,
) -> None:
    h_patches, w_patches = sample.patch_grid
    expected_patches = h_patches * w_patches
    if patch_rgb.shape != (expected_patches, 3):
        raise ValueError(
            f"Patch RGB values must have shape ({expected_patches}, 3) for "
            f"{h_patches}x{w_patches} patches, got {tuple(patch_rgb.shape)}."
        )

    patch_colors = patch_rgb.view(h_patches, w_patches, 3).permute(2, 0, 1)
    patch_colors = patch_colors.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    original = sample.image_resized.clamp(0.0, 1.0)
    if patch_colors.shape != original.shape:
        raise ValueError(
            f"Patch color image has shape {tuple(patch_colors.shape)}, expected {tuple(original.shape)}."
        )

    overlay = original.mul(1.0 - alpha).add(patch_colors.mul(alpha)).clamp(0.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(overlay).save(output_path)


def save_original_rgb_with_patch_alpha_image(
    patch_alpha: torch.Tensor,
    sample: ImageNetSample,
    output_path: Path,
    *,
    patch_size: int,
) -> None:
    h_patches, w_patches = sample.patch_grid
    expected_patches = h_patches * w_patches
    if patch_alpha.shape != (expected_patches,):
        raise ValueError(
            f"Patch alpha values must have shape ({expected_patches},) for "
            f"{h_patches}x{w_patches} patches, got {tuple(patch_alpha.shape)}."
        )

    alpha = patch_alpha.view(h_patches, w_patches).unsqueeze(0)
    alpha = alpha.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    original = sample.image_resized.clamp(0.0, 1.0)
    if alpha.shape[-2:] != original.shape[-2:]:
        raise ValueError(
            f"Alpha image has shape {tuple(alpha.shape)}, expected spatial shape {tuple(original.shape[-2:])}."
        )

    rgba = torch.cat((original, alpha.clamp(0.0, 1.0)), dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(rgba).save(output_path)


def pc_overlay_color(component_index: int) -> tuple[str, torch.Tensor]:
    if component_index < len(PC_OVERLAY_COLORS):
        color_name, color = PC_OVERLAY_COLORS[component_index]
    else:
        hue = (0.08 + (component_index - len(PC_OVERLAY_COLORS)) * 0.61803398875) % 1.0
        color_name = f"hue_{int(round(hue * 360.0)):03d}"
        color = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return color_name, torch.tensor(color, dtype=torch.float32)


def pc_component_values(projection: PCA3DProjection, component_index: int, *, flipped: bool) -> torch.Tensor:
    if component_index < 0 or component_index >= projection.coordinates.shape[1]:
        raise ValueError(
            f"Component index {component_index} is outside PCA component range [0, "
            f"{projection.coordinates.shape[1] - 1}]."
        )
    if not flipped:
        return projection.component_values[:, component_index]
    coordinates = projection.coordinates[:, component_index]
    coordinates = -coordinates
    return torch.sigmoid(coordinates.mul(2.0)).clamp(0.0, 1.0)


def pc_patch_rgb(projection: PCA3DProjection, component_index: int, *, flipped: bool) -> tuple[str, torch.Tensor]:
    color_name, color = pc_overlay_color(component_index)
    values = pc_component_values(projection, component_index, flipped=flipped).unsqueeze(1)
    return color_name, values.mul(color.unsqueeze(0))


def rgb_triplet_values(projection: PCA3DProjection, component_indices: tuple[int, int, int]) -> torch.Tensor:
    max_index = projection.component_values.shape[1] - 1
    invalid_indices = [component_index for component_index in component_indices if component_index > max_index]
    if invalid_indices:
        raise ValueError(
            f"RGB triplet overlay requested 0-based PCA component indices {invalid_indices}, "
            f"but only components 0 through {max_index} are available."
        )
    return projection.component_values[:, list(component_indices)]


def image_output_dir(output_dir: Path, sample: ImageNetSample) -> Path:
    return output_dir / f"imagenet_val_{sample.index:05d}"


def save_original_image(sample: ImageNetSample, image_dir: Path) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    output_path = image_dir / "original.jpg"
    if sample.raw_image is not None:
        sample.raw_image.save(output_path)
    else:
        TF.to_pil_image(sample.image_resized.clamp(0.0, 1.0)).save(output_path)
    return output_path


def run_for_checkpoint(
    *,
    checkpoint_path: Path,
    config_file: Path,
    sample: ImageNetSample,
    image_dir: Path,
    device: torch.device,
    point_size: float,
    alpha: float,
    patch_color_alpha: float,
    n_eigen_vec: int,
    elev: float,
    azim: float,
    plot_width: int,
    plot_height: int,
    plotly_js: str,
) -> None:
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

    patch_features = extract_final_patch_features(
        model,
        sample.image,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_pca_components = max(n_eigen_vec, MIN_RGB_TRIPLET_COMPONENTS)
    projection = compute_pca_3d_projection(patch_features, n_components=n_pca_components)
    output_stem = safe_output_stem(model_name, checkpoint_path)
    patch_size_pixels = sample.image_resized.shape[-2] // sample.patch_grid[0]
    figure_path = image_dir / "html" / f"{output_stem}.html"
    save_pca_3d_html(
        projection,
        figure_path,
        model_name=model_name,
        sample=sample,
        point_size=point_size,
        alpha=alpha,
        elev=elev,
        azim=azim,
        plot_width=plot_width,
        plot_height=plot_height,
        plotly_js=plotly_js,
    )
    patch_image_path = image_dir / "patch_overlay" / f"{output_stem}.png"
    save_patch_colored_image(
        projection.rgb,
        sample,
        patch_image_path,
        patch_size=patch_size_pixels,
        alpha=patch_color_alpha,
    )
    patch_overlay_paths = [patch_image_path]

    for output_dirname, overlay_alpha in (
        ("patch_overlay_rgb_alpha_1_0", 1.0),
        (
            f"patch_overlay_rgb_alpha_{TRANSPARENT_PATCH_COLOR_ALPHA:g}".replace(".", "_"),
            TRANSPARENT_PATCH_COLOR_ALPHA,
        ),
    ):
        extra_patch_image_path = image_dir / output_dirname / f"{output_stem}.png"
        save_patch_colored_image(
            projection.rgb,
            sample,
            extra_patch_image_path,
            patch_size=patch_size_pixels,
            alpha=overlay_alpha,
        )
        patch_overlay_paths.append(extra_patch_image_path)

    for output_dirname, component_indices in RGB_TRIPLET_OVERLAYS:
        rgb_triplet_patch_image_path = (
            image_dir / "patch_overlay_rgb_eigenvectors" / output_dirname / f"{output_stem}.png"
        )
        save_patch_colored_image(
            rgb_triplet_values(projection, component_indices),
            sample,
            rgb_triplet_patch_image_path,
            patch_size=patch_size_pixels,
            alpha=1.0,
        )
        patch_overlay_paths.append(rgb_triplet_patch_image_path)

    for component_index in range(n_eigen_vec):
        color_name, component_rgb = pc_patch_rgb(projection, component_index, flipped=False)
        component_dirname = f"pc{component_index + 1:02d}_{color_name}"
        component_patch_image_path = (
            image_dir / "patch_overlay_eigenvectors" / component_dirname / f"{output_stem}.png"
        )
        save_patch_colored_image(
            component_rgb,
            sample,
            component_patch_image_path,
            patch_size=patch_size_pixels,
            alpha=1.0,
        )
        patch_overlay_paths.append(component_patch_image_path)

        _, flipped_component_rgb = pc_patch_rgb(projection, component_index, flipped=True)
        flipped_component_patch_image_path = (
            image_dir / "patch_overlay_eigenvectors_flipped" / component_dirname / f"{output_stem}.png"
        )
        save_patch_colored_image(
            flipped_component_rgb,
            sample,
            flipped_component_patch_image_path,
            patch_size=patch_size_pixels,
            alpha=1.0,
        )
        patch_overlay_paths.append(flipped_component_patch_image_path)

    for output_dirname, channel_index in PC_ALPHA_IMAGES:
        rgba_patch_image_path = image_dir / output_dirname / f"{output_stem}.png"
        save_original_rgb_with_patch_alpha_image(
            projection.component_values[:, channel_index],
            sample,
            rgba_patch_image_path,
            patch_size=patch_size_pixels,
        )
        patch_overlay_paths.append(rgba_patch_image_path)

    logger.info("Wrote %s", figure_path.resolve())
    for patch_overlay_path in patch_overlay_paths:
        logger.info("Wrote %s", patch_overlay_path.resolve())


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.point_alpha <= 0 or args.point_alpha > 1:
        raise ValueError("--point-alpha/--alpha must be in the interval (0, 1].")
    if args.patch_color_alpha < 0 or args.patch_color_alpha > 1:
        raise ValueError("--patch-color-alpha must be in the interval [0, 1].")
    if args.n_eigen_vec < 3:
        raise ValueError("--n-eigen-vec must be at least 3.")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive.")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if args.image_size % args.patch_size != 0:
        raise ValueError("--image-size must be divisible by --patch-size.")
    if args.plot_width <= 0 or args.plot_height <= 0:
        raise ValueError("--plot-width and --plot-height must be positive.")

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

    image_dir = image_output_dir(args.output_dir, sample)
    input_path = save_original_image(sample, image_dir)
    logger.info("Wrote %s", input_path.resolve())

    device = torch.device(args.device)
    for checkpoint_path, config_file in checkpoint_specs:
        run_for_checkpoint(
            checkpoint_path=checkpoint_path,
            config_file=config_file,
            sample=sample,
            image_dir=image_dir,
            device=device,
            point_size=args.point_size,
            alpha=args.point_alpha,
            patch_color_alpha=args.patch_color_alpha,
            n_eigen_vec=args.n_eigen_vec,
            elev=args.elev,
            azim=args.azim,
            plot_width=args.plot_width,
            plot_height=args.plot_height,
            plotly_js=args.plotly_js,
        )

    logger.info("Wrote 3D PCA visualizations to %s", image_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
