from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from scipy import signal
from sklearn.decomposition import PCA

from dinov3.configs.config import get_default_config
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval


logger = logging.getLogger("measurements.pca_dino")

DEFAULT_MODEL_DIRS = (
    "vits_claude",
    "vits_1L_claude",
    "vits_3L_claude",
    "vits_5L_claude",
    "vits_6L_claude",
)
DEFAULT_IMAGE_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DINOv3 PCA patch visualizations on local DINO checkpoints."
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        type=Path,
        default=[Path(model_dir) for model_dir in DEFAULT_MODEL_DIRS],
        help="Model run directories containing config.yaml and eval/training_*/teacher_checkpoint.pth.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=124999,
        help="Training step to load from each model directory.",
    )
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
        "--foreground-classifier",
        type=Path,
        default=Path("fg_classifier.pkl"),
        help="Pickle containing the foreground classifier used by notebooks/pca.ipynb.",
    )
    parser.add_argument(
        "--foreground-mask",
        type=Path,
        default=None,
        help=(
            "Saved foreground mask or score tensor. If provided, this is thresholded with "
            "--foreground-threshold and used instead of --foreground-classifier."
        ),
    )
    parser.add_argument(
        "--no-foreground-filter",
        action="store_true",
        help="Fit PCA on all patches and do not mask the visualization.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_URI,
        help="Input image path or URL. Defaults to the notebook test image.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("measurements") / "dino_pca_results")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--median-kernel-size", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--save-foreground-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save the notebook's image/foreground-score diagnostic figure.",
    )
    parser.add_argument(
        "--save-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a comparison grid containing the input image and all PCA visualizations.",
    )
    return parser.parse_args()


def prepare_matplotlib_cache() -> None:
    cache_root = Path(f"/tmp/pca_dino_plot_cache_{os.getuid()}")
    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib")))
    xdg_cache_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg")))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)


def infer_config_file(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    for parent in checkpoint_path.parents:
        config_file = parent / "config.yaml"
        if config_file.is_file():
            return config_file
    raise FileNotFoundError(f"Could not infer config.yaml from checkpoint path: {checkpoint_path}")


def infer_checkpoint_step(checkpoint_path: Path) -> int | None:
    for part in checkpoint_path.parts:
        match = re.fullmatch(r"training_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def infer_model_name(config: Any, config_file: Path) -> str:
    wandb_name = OmegaConf.select(config, "wandb.name")
    if wandb_name:
        return str(wandb_name)
    return config_file.parent.name


def load_config(config_file: Path):
    return OmegaConf.merge(get_default_config(), OmegaConf.load(config_file))


def checkpoint_from_model_dir(model_dir: Path, checkpoint_step: int) -> Path:
    return model_dir / "eval" / f"training_{checkpoint_step}" / "teacher_checkpoint.pth"


def resolve_checkpoint_specs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.checkpoints is None:
        specs = []
        for model_dir in args.model_dirs:
            checkpoint = checkpoint_from_model_dir(model_dir, args.checkpoint_step)
            specs.append((checkpoint, model_dir / "config.yaml"))
        return specs

    if args.config_files is not None and len(args.config_files) != len(args.checkpoints):
        raise ValueError("--config-files must have the same length as --checkpoints.")

    specs = []
    for idx, checkpoint in enumerate(args.checkpoints):
        config_file = args.config_files[idx] if args.config_files is not None else infer_config_file(checkpoint)
        specs.append((checkpoint, config_file))
    return specs


def load_image(image_path_or_url: str) -> Image.Image:
    if re.match(r"^https?://", image_path_or_url):
        with urllib.request.urlopen(image_path_or_url) as response:
            return Image.open(response).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def resize_transform(image: Image.Image, *, image_size: int, patch_size: int) -> torch.Tensor:
    width, height = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((width * image_size) / (height * patch_size))
    return TF.to_tensor(TF.resize(image, (h_patches * patch_size, w_patches * patch_size)))


def get_num_layers(model: torch.nn.Module) -> int:
    if hasattr(model, "n_blocks"):
        return int(model.n_blocks)
    if hasattr(model, "blocks"):
        return len(model.blocks)
    raise AttributeError("Expected the model to expose n_blocks or blocks for notebook-style layer selection.")


def autocast_context(device: torch.device, autocast_dtype: torch.dtype):
    if device.type == "cuda" and autocast_dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)
    return nullcontext()


@torch.inference_mode()
def extract_final_patch_features(
    model: torch.nn.Module,
    image_resized_norm: torch.Tensor,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    n_layers = get_num_layers(model)
    image_batch = image_resized_norm.unsqueeze(0).to(device, non_blocking=True)
    with autocast_context(device, autocast_dtype):
        feats = model.get_intermediate_layers(image_batch, n=range(n_layers), reshape=True, norm=True)
    final_features = feats[-1].squeeze().detach().cpu()
    dim = final_features.shape[0]
    return final_features.view(dim, -1).permute(1, 0).float()


def predict_foreground_scores(
    classifier,
    patch_features: torch.Tensor,
    h_patches: int,
    w_patches: int,
    *,
    median_kernel_size: int,
) -> torch.Tensor:
    try:
        foreground_probs = classifier.predict_proba(patch_features.numpy())[:, 1]
    except ValueError as exc:
        raise ValueError(
            "Foreground classifier is incompatible with these patch features. "
            f"The extracted feature dimension is {patch_features.shape[1]}; "
            "use a classifier trained on features from the same model family/checkpoint."
        ) from exc
    fg_score = foreground_probs.reshape(h_patches, w_patches)
    return torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=median_kernel_size))


def foreground_mask_from_scores(fg_score_mf: torch.Tensor, *, foreground_threshold: float) -> torch.Tensor:
    return fg_score_mf.view(-1) > foreground_threshold


def load_foreground_mask(mask_path: Path, h_patches: int, w_patches: int, *, foreground_threshold: float) -> torch.Tensor:
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing foreground mask: {mask_path}")

    loaded = torch.load(mask_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        for key in ("fg_score_mf", "fg_score", "foreground_score", "foreground_mask", "mask"):
            if key in loaded:
                loaded = loaded[key]
                break
        else:
            raise ValueError(
                f"Foreground mask file {mask_path} contains a dict, but none of the expected keys were found: "
                "fg_score_mf, fg_score, foreground_score, foreground_mask, mask."
            )

    mask_or_score = torch.as_tensor(loaded).detach().cpu()
    mask_or_score = mask_or_score.squeeze()
    expected_numel = h_patches * w_patches

    if mask_or_score.shape == (expected_numel,):
        mask_or_score = mask_or_score.view(h_patches, w_patches)
    elif mask_or_score.shape != (h_patches, w_patches):
        raise ValueError(
            f"Foreground mask {mask_path} has shape {tuple(mask_or_score.shape)}, expected "
            f"({h_patches}, {w_patches}) or ({expected_numel},)."
        )

    if mask_or_score.dtype == torch.bool:
        return mask_or_score
    return mask_or_score.float() > foreground_threshold


def compute_projected_image(
    patch_features: torch.Tensor,
    *,
    h_patches: int,
    w_patches: int,
    foreground_mask: torch.Tensor | None,
) -> torch.Tensor:
    if foreground_mask is None:
        patch_selection = torch.ones(patch_features.shape[0], dtype=torch.bool)
    else:
        patch_selection = foreground_mask.view(-1).bool()
        if patch_selection.numel() != patch_features.shape[0]:
            raise ValueError(
                f"Foreground mask contains {patch_selection.numel()} patches, but extracted "
                f"{patch_features.shape[0]} patch features."
            )

    if not bool(patch_selection.any()):
        raise RuntimeError("No patches were selected for PCA. Check the foreground mask or lower --foreground-threshold.")
    if int(patch_selection.sum().item()) < 3:
        raise RuntimeError("PCA visualization needs at least 3 selected patches.")

    foreground_patches = patch_features[patch_selection]
    pca = PCA(n_components=3, whiten=True)
    pca.fit(foreground_patches.numpy())

    projected_image = torch.from_numpy(pca.transform(patch_features.numpy())).view(h_patches, w_patches, 3)
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
    if foreground_mask is not None:
        projected_image *= foreground_mask.unsqueeze(0)
    return projected_image


def save_foreground_score_figure(
    image: Image.Image,
    fg_score_mf: torch.Tensor,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "axes.labelsize": 5,
            "axes.titlesize": 4,
        }
    )
    fig = plt.figure(figsize=(4, 2), dpi=dpi)
    ax_image = fig.add_subplot(1, 2, 1)
    ax_image.imshow(image)
    ax_image.axis("off")
    ax_image.set_title(f"Image, Size {image.size}")

    ax_score = fig.add_subplot(1, 2, 2)
    score_image = ax_score.imshow(fg_score_mf)
    ax_score.set_title(f"Foreground Score, Size {tuple(fg_score_mf.shape)}")
    fig.colorbar(score_image, ax=ax_score)
    ax_score.axis("off")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_pca_figure(projected_image: torch.Tensor, output_path: Path, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(projected_image.permute(1, 2, 0).numpy())
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_grid_figure(image: Image.Image, pca_images: dict[str, torch.Tensor], output_path: Path, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    ncols = len(pca_images) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols, 2.4), dpi=dpi)
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(image)
    axes[0].set_title("Image", fontsize=8)
    axes[0].axis("off")

    for ax, (model_name, projected_image) in zip(axes[1:], pca_images.items()):
        ax.imshow(projected_image.permute(1, 2, 0).numpy())
        ax.set_title(model_name, fontsize=8)
        ax.axis("off")

    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def safe_output_stem(model_name: str, checkpoint_path: Path) -> str:
    checkpoint_step = infer_checkpoint_step(checkpoint_path)
    suffix = f"_training_{checkpoint_step}" if checkpoint_step is not None else ""
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return f"{safe_model_name}{suffix}"


def run_for_checkpoint(
    *,
    checkpoint_path: Path,
    config_file: Path,
    classifier,
    foreground_mask: torch.Tensor | None,
    use_foreground_classifier: bool,
    image: Image.Image,
    image_resized_norm: torch.Tensor,
    h_patches: int,
    w_patches: int,
    output_dir: Path,
    device: torch.device,
    foreground_threshold: float,
    median_kernel_size: int,
    dpi: int,
    save_foreground_score: bool,
) -> tuple[str, torch.Tensor]:
    checkpoint_path = checkpoint_path.resolve()
    config_file = config_file.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing config file: {config_file}")

    config = load_config(config_file)
    model_name = infer_model_name(config, config_file)
    logger.info("Loading %s from %s", model_name, checkpoint_path)
    model = build_model_for_eval(config, checkpoint_path)
    model.to(device)
    model.eval()

    autocast_dtype = get_autocast_dtype(config)
    patch_features = extract_final_patch_features(
        model,
        image_resized_norm,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fg_score_mf = None
    if foreground_mask is not None:
        checkpoint_foreground_mask = foreground_mask
    elif use_foreground_classifier:
        fg_score_mf = predict_foreground_scores(
            classifier,
            patch_features,
            h_patches,
            w_patches,
            median_kernel_size=median_kernel_size,
        )
        checkpoint_foreground_mask = foreground_mask_from_scores(
            fg_score_mf,
            foreground_threshold=foreground_threshold,
        ).view(h_patches, w_patches)
    else:
        checkpoint_foreground_mask = None

    projected_image = compute_projected_image(
        patch_features,
        h_patches=h_patches,
        w_patches=w_patches,
        foreground_mask=checkpoint_foreground_mask,
    )

    output_stem = safe_output_stem(model_name, checkpoint_path)
    if save_foreground_score and fg_score_mf is not None:
        save_foreground_score_figure(
            image,
            fg_score_mf,
            output_dir / f"{output_stem}_foreground_score.png",
            dpi=dpi,
        )
    save_pca_figure(projected_image, output_dir / f"{output_stem}_pca.png", dpi=dpi)
    return output_stem, projected_image


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_matplotlib_cache()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.median_kernel_size <= 0 or args.median_kernel_size % 2 == 0:
        raise ValueError("--median-kernel-size must be a positive odd integer.")
    if args.no_foreground_filter and args.foreground_mask is not None:
        raise ValueError("--no-foreground-filter cannot be combined with --foreground-mask.")

    use_foreground_classifier = not args.no_foreground_filter and args.foreground_mask is None
    if use_foreground_classifier and not args.foreground_classifier.is_file():
        raise FileNotFoundError(
            f"Missing foreground classifier: {args.foreground_classifier}. "
            "This script intentionally follows notebooks/pca.ipynb, which loads fg_classifier.pkl. "
            "Pass --foreground-classifier with a classifier trained for the target checkpoint features, "
            "or use --foreground-mask / --no-foreground-filter."
        )

    classifier = None
    if use_foreground_classifier:
        with args.foreground_classifier.open("rb") as f:
            classifier = pickle.load(f)

    image = load_image(args.image)
    image_resized = resize_transform(image, image_size=args.image_size, patch_size=args.patch_size)
    image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    h_patches, w_patches = [int(d / args.patch_size) for d in image_resized.shape[1:]]
    foreground_mask = None
    if args.foreground_mask is not None:
        foreground_mask = load_foreground_mask(
            args.foreground_mask,
            h_patches,
            w_patches,
            foreground_threshold=args.foreground_threshold,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    pca_images: dict[str, torch.Tensor] = {}
    for checkpoint_path, config_file in resolve_checkpoint_specs(args):
        model_key, projected_image = run_for_checkpoint(
            checkpoint_path=checkpoint_path,
            config_file=config_file,
            classifier=classifier,
            foreground_mask=foreground_mask,
            use_foreground_classifier=use_foreground_classifier,
            image=image,
            image_resized_norm=image_resized_norm,
            h_patches=h_patches,
            w_patches=w_patches,
            output_dir=args.output_dir,
            device=device,
            foreground_threshold=args.foreground_threshold,
            median_kernel_size=args.median_kernel_size,
            dpi=args.dpi,
            save_foreground_score=args.save_foreground_score,
        )
        pca_images[model_key] = projected_image

    if args.save_grid and pca_images:
        save_grid_figure(image, pca_images, args.output_dir / "pca_grid.png", dpi=args.dpi)

    logger.info("Wrote PCA visualizations to %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
