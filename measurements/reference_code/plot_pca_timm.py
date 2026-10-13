#!/usr/bin/env python3
"""
PCA Visualization Script for timm Models

Performs PCA visualization on models from:
- processed_artifacts_kP.json (ImageNet)
- cifar10_aggregated_results.json (CIFAR10)
- cifar100_aggregated_results.json (CIFAR100)

For each model and drop path rate, selects the best performing seed
and generates PCA projection plots.

Output structure:
    pca_plots_timm/{dataset}/{model_name}/drop_path={dp}/{layer}.pdf
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from timm.models import create_model
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

# Matplotlib configuration for publication-quality plots
mpl.rcParams.update({
    "mathtext.fontset": "cm",
    "mathtext.rm": "CMU Serif",
    "mathtext.it": "CMU Serif:italic",
    "mathtext.bf": "CMU Serif:bold",
    "font.family": "serif",
})

# ============================================================================
#                           CONFIGURATION
# ============================================================================

# Default paths (can be overridden via CLI)
DEFAULT_IMAGENET_JSON = "processed_artifacts_kP.json"
DEFAULT_CIFAR10_JSON = "cifar10_aggregated_results.json"
DEFAULT_CIFAR100_JSON = "cifar100_aggregated_results.json"

# Checkpoint base directories
IMAGENET_CHECKPOINT_BASE = "/home/yucz/scratch/timm_imagenet"
CIFAR10_CHECKPOINT_BASE = "/home/yucz/scratch/timm_cifar10"
CIFAR100_CHECKPOINT_BASE = "/home/yucz/scratch/timm_cifar100"

# Dataset root directories
DEFAULT_IMAGENET_ROOT = "/datasets/imagenet"
DEFAULT_CIFAR10_ROOT = "/home/yucz/scratch/cifar10"
DEFAULT_CIFAR100_ROOT = "/home/yucz/scratch/cifar100"

# Output directory
DEFAULT_OUTPUT_DIR = "pca_plots_timm"

# ============================================================================
#                           HELPER FUNCTIONS
# ============================================================================


def simplify_model_name(m_name: str) -> str:
    """Convert model name to a human-readable simplified version."""
    if "one_P_head" in m_name:
        return "Proposed: 1 P-head"
    elif "three_P_head" in m_name:
        return "Proposed: 3 P-heads"
    elif "mix_depth" in m_name:
        return "Proposed: mixing depth-wise"
    elif "facebook" in m_name:
        return "Baseline"
    elif "all_I_minus" in m_name:
        return "Proposed: all Laplacians"
    elif "deit3_large_4P" in m_name:
        return "Proposed: 4 P-heads"
    elif "deit3_base_3P" in m_name:
        return "Proposed: 3 P-heads"
    elif "deit3_base_6P" in m_name:
        return "Proposed: 6 P-heads"
    elif "deit3_base_1P" in m_name:
        return "Proposed: 1 P-head"
    elif "deit3_base_9P" in m_name:
        return "Proposed: 9 P-heads"
    elif "deit3_base_0P" in m_name:
        return "Proposed: 0 P-heads"
    elif "deit3_large" in m_name or "deit3_base_qk" in m_name:
        return "Baseline"
    elif "interleave_attn_first" in m_name:
        return "Interleave: P first"
    elif "interleave_attn_last" in m_name:
        return "Interleave: P last"
    else:
        # Return the original name if no pattern matches
        return m_name


def simplify_layer_name(layer_name: str) -> str:
    """Convert layer name to a human-readable format."""
    if layer_name.startswith("block_") and layer_name[6:].isdigit():
        return f"Block {layer_name[6:]}"
    elif layer_name == "norm_final":
        return "Final LayerNorm"
    else:
        return layer_name


def strip_compile_prefix(state_dict: Dict) -> Dict:
    """
    Remove '_orig_mod.' prefix from state dict keys.

    Models saved after torch.compile() have this prefix added to all keys.
    This function strips it to allow loading into non-compiled models.
    """
    prefix = "_orig_mod."
    # Check if any key has the prefix
    has_prefix = any(k.startswith(prefix) for k in state_dict.keys())

    if not has_prefix:
        return state_dict

    # Strip the prefix from all keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


# ============================================================================
#                           PCA COMPUTATION
# ============================================================================


def compute_pca_projection(x: torch.Tensor) -> torch.Tensor:
    """
    Compute PCA projection to 2D for visualization.

    Args:
        x: Tensor of shape (B, T, C) where B=batch, T=tokens, C=channels

    Returns:
        Tensor of shape (B, T, 2) with 2D projections
    """
    assert len(x.size()) == 3
    B, T, C = x.shape
    x = x - x.mean(dim=(0, 1))  # Center before PCA
    U, S, Vh = torch.linalg.svd(x.view(-1, C))  # Use all tokens for PCA
    top2_eigenvec = Vh[:2, :]
    projection = torch.matmul(x, top2_eigenvec.transpose(0, 1))  # (B, T, 2)
    return projection


class PCAHook:
    """Hook to capture and compute PCA projections at each layer."""

    def __init__(self):
        self.projection_per_layer = {}
        self.handles = []

    def _pca_hook(self, layer_name: str):
        def fn(m, input, output):
            x = output
            projections = compute_pca_projection(x)
            self.projection_per_layer[layer_name] = projections
        return fn

    def register(self, model):
        """Register hooks on model blocks and final norm."""
        for idx, block in enumerate(model.blocks):
            self.handles.append(
                block.register_forward_hook(self._pca_hook(f"block_{idx}"))
            )
        self.handles.append(
            model.norm.register_forward_hook(self._pca_hook("norm_final"))
        )

    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()


# ============================================================================
#                           DATA LOADING
# ============================================================================


def extract_labels(ds):
    """Extract labels from various dataset types."""
    if hasattr(ds, "targets") and ds.targets is not None:
        return list(ds.targets)
    if hasattr(ds, "labels") and ds.labels is not None:
        return list(ds.labels)
    if hasattr(ds, "samples"):
        return [y for _, y in ds.samples]
    if hasattr(ds, "_labels"):
        return ds._labels
    raise AttributeError("Could not find labels/targets on the dataset.")


def get_transform(size: int):
    """Get standard transform for all datasets."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def build_balanced_batch(
    dataset,
    selected_classes: List[int],
    n_samples_per_class: int,
    seed: Optional[int] = None,
    num_workers: int = 8,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a balanced batch by sampling n indices per selected class.

    Args:
        dataset: Dataset with .targets or .samples attribute
        selected_classes: List of class indices to sample from
        n_samples_per_class: Number of samples per class
        seed: Random seed for reproducibility
        num_workers: Number of data loader workers
        batch_size: Batch size for loading
        device: Device to move tensors to

    Returns:
        Tuple of (images, labels) tensors
    """
    print(f"Creating balanced batch with {len(selected_classes)} classes, "
          f"{n_samples_per_class} samples per class")

    # Get targets
    if hasattr(dataset, "targets"):
        targets = np.asarray(dataset.targets)
    elif hasattr(dataset, "samples"):
        targets = np.asarray([cls for _, cls in dataset.samples])
    else:
        raise AttributeError("Dataset must expose .targets or .samples")

    sel = np.asarray(list(map(int, selected_classes)), dtype=np.int64)

    # Sample indices per class
    rng = np.random.default_rng(seed)
    selected_indices = []
    for c in sel:
        pool = np.flatnonzero(targets == c)
        if pool.size < n_samples_per_class:
            raise ValueError(
                f"Class {c} has only {pool.size} samples, "
                f"need {n_samples_per_class}"
            )
        take = rng.choice(pool, size=n_samples_per_class, replace=False)
        selected_indices.append(take)

    indices = np.concatenate(selected_indices)
    rng.shuffle(indices)
    indices = indices.tolist()

    # Load data
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    imgs_list, lbls_list = [], []
    for imgs, lbls in loader:
        imgs_list.append(imgs)
        lbls_list.append(lbls)

    images = torch.cat(imgs_list, dim=0)
    labels = torch.cat(lbls_list, dim=0)

    if device is not None:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

    return images, labels


# ============================================================================
#                           PLOTTING
# ============================================================================


Color = Union[str, tuple]


def _color_for_label(label: int, s: float = 0.65, v: float = 0.9):
    """Generate deterministic color for a label using golden ratio."""
    phi_conj = (5 ** 0.5 - 1) / 2
    h = (label * phi_conj) % 1.0
    return mcolors.hsv_to_rgb((h, s, v))


def plot_sequences_by_label(
    projections: torch.Tensor,
    labels: torch.Tensor,
    *,
    ax=None,
    s: float = 10,
    alpha: float = 0.8,
    marker: str = "o",
    legend: bool = False,
    title: Optional[str] = None,
    figsize=(6, 6),
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    bbox_inches: str = "tight",
    transparent: bool = False,
    close: bool = True,
    color_map: Optional[Mapping[int, Color]] = None,
    clip_quantile: Optional[float] = None,
):
    """
    Plot 2D PCA projections colored by class label.

    Args:
        projections: (B, T, 2) tensor of 2D projections
        labels: (B,) tensor of class labels
        ax: Optional matplotlib axes
        s, alpha, marker: Scatter plot styling
        legend: Whether to show legend
        title: Plot title
        figsize: Figure size
        out_path: Path to save figure (as PDF)
        dpi: DPI for saving
        bbox_inches: Bounding box option
        transparent: Whether to use transparent background
        close: Whether to close figure after saving
        color_map: Optional custom color mapping
        clip_quantile: If set (e.g., 0.9), constrain xlim/ylim so only
            points within this quantile of magnitude are fully visible.
            Points outside this range will still be plotted but may be
            clipped by the axis limits.

    Returns:
        (fig, ax) tuple
    """
    if projections.ndim != 3 or projections.shape[-1] != 2:
        raise ValueError("projections must have shape (B, T, 2)")
    B, T, _ = projections.shape
    if labels.shape != (B,):
        raise ValueError(f"labels must have shape (B,), got {labels.shape}")

    P = projections.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        created_fig = True
    else:
        fig = ax.figure

    classes = np.unique(y)
    for c in classes:
        mask = (y == c)
        pts = P[mask].reshape(-1, 2)
        if pts.size == 0:
            continue
        color = (color_map[c] if (color_map is not None and c in color_map)
                 else _color_for_label(int(c)))
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=s, alpha=alpha,
                   marker=marker, label=str(int(c)))

    # Apply quantile-based axis limits if requested
    if clip_quantile is not None:
        all_pts = P.reshape(-1, 2)
        magnitudes = np.sqrt(all_pts[:, 0]**2 + all_pts[:, 1]**2)
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
    if legend:
        ax.legend(title="label", frameon=False, markerscale=1.5)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches,
                    transparent=transparent, format='pdf')

    if close and (created_fig or out_path is not None):
        plt.close(fig)

    return fig, ax


# ============================================================================
#                       JSON PARSING & CHECKPOINT RESOLUTION
# ============================================================================


def parse_imagenet_json(json_path: str) -> List[Dict]:
    """
    Parse ImageNet JSON file and return list of model configurations.

    Returns list of dicts with keys:
        - model_name: Full model name for timm
        - drop_path: Drop path rate
        - best_seed: Best performing seed
        - best_top1: Best top-1 accuracy
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []
    detailed = data.get('detailed', {})

    for key, entry in detailed.items():
        model_name = entry['model_name']
        drop_path = entry['drop_path']
        per_seed = entry.get('per_seed_results', [])

        if not per_seed:
            continue

        # Find best performing seed
        best_result = max(per_seed, key=lambda x: x['best_eval_top1'])

        results.append({
            'model_name': model_name,
            'drop_path': drop_path,
            'best_seed': best_result['seed'],
            'best_top1': best_result['best_eval_top1'],
        })

    return results


def parse_cifar_json(json_path: str, dataset: str) -> List[Dict]:
    """
    Parse CIFAR10/100 JSON file and return list of model configurations.

    Args:
        json_path: Path to JSON file
        dataset: "cifar10" or "cifar100"

    Returns list of dicts with keys:
        - model_name: Full model name for timm
        - drop_path: Drop path rate
        - best_seed: Best performing seed
        - best_top1: Best top-1 accuracy
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []
    models = data.get('models', {})

    for model_name, dp_data in models.items():
        for dp_str, dp_entry in dp_data.items():
            drop_path = float(dp_str)
            seeds = dp_entry.get('seeds', [])
            individual_top1 = dp_entry.get('individual_best_top1', [])

            if not seeds or not individual_top1:
                continue

            # Find best performing seed
            best_idx = np.argmax(individual_top1)
            best_seed = seeds[best_idx]
            best_top1 = individual_top1[best_idx]

            results.append({
                'model_name': model_name,
                'drop_path': drop_path,
                'best_seed': best_seed,
                'best_top1': best_top1,
            })

    return results


def get_imagenet_checkpoint_path(
    model_name: str,
    drop_path: float,
    seed: int,
    base_dir: str = IMAGENET_CHECKPOINT_BASE
) -> Optional[str]:
    """
    Construct checkpoint path for ImageNet model.

    Handles multiple directory naming conventions:
    - drop_path=0.4: {model}_patch16_224/{model}_patch16_224_in1k_seed={seed}/
    - other drop_path: {model}_patch16_224/{model}_patch16_224_in1k_seed={seed}_drop_path={dp}/
    """
    # The model name in the JSON is like "deit3_base_0P_qk_norm"
    # The actual timm model name and directory might be "deit3_base_0P_qk_norm_patch16_224"
    model_with_suffix = f"{model_name}_patch16_224"

    # Build list of patterns to try (in order of specificity)
    patterns = []

    # For drop_path=0.4 (default), try without drop_path in dir name first
    if abs(drop_path - 0.4) < 0.01:
        patterns.append(
            (model_with_suffix, f"{model_with_suffix}_in1k_seed={seed}")
        )

    # Pattern with explicit drop_path in dir name
    patterns.append(
        (model_with_suffix, f"{model_with_suffix}_in1k_seed={seed}_drop_path={drop_path}")
    )

    # Also try without drop_path for other values (fallback)
    if abs(drop_path - 0.4) >= 0.01:
        patterns.append(
            (model_with_suffix, f"{model_with_suffix}_in1k_seed={seed}")
        )

    # Try run-style naming as last resort
    patterns.append(
        (model_with_suffix, f"{model_with_suffix}_in1k_run1")
    )

    for model_dir, run_dir in patterns:
        ckpt_path = os.path.join(base_dir, model_dir, run_dir, "model_best.pth.tar")
        if os.path.exists(ckpt_path):
            return ckpt_path

    # Also try without _patch16_224 suffix if model_name already has it
    if "_patch16_224" in model_name:
        alt_patterns = [
            (model_name, f"{model_name}_in1k_seed={seed}"),
            (model_name, f"{model_name}_in1k_seed={seed}_drop_path={drop_path}"),
        ]
        for model_dir, run_dir in alt_patterns:
            ckpt_path = os.path.join(base_dir, model_dir, run_dir, "model_best.pth.tar")
            if os.path.exists(ckpt_path):
                return ckpt_path

    return None


def get_cifar_checkpoint_path(
    model_name: str,
    drop_path: float,
    seed: int,
    dataset: str,
    base_dir: Optional[str] = None
) -> Optional[str]:
    """
    Construct checkpoint path for CIFAR10/100 model.

    Directory pattern: {model}_seed{seed}_dp{dp}/best.pth
    """
    if base_dir is None:
        base_dir = CIFAR10_CHECKPOINT_BASE if dataset == "cifar10" else CIFAR100_CHECKPOINT_BASE

    # Directory naming: deit3_base_0P_qk_norm_patch4_32_seed42_dp0.1
    run_dir = f"{model_name}_seed{seed}_dp{drop_path}"
    ckpt_path = os.path.join(base_dir, run_dir, "best.pth")

    if os.path.exists(ckpt_path):
        return ckpt_path

    return None


def get_timm_model_name(model_name: str, dataset: str) -> str:
    """
    Get the timm model name from the stored model name.

    ImageNet models typically need _patch16_224 suffix.
    CIFAR models typically have _patch4_32 suffix already.
    """
    if dataset == "imagenet":
        if "_patch16_224" not in model_name:
            return f"{model_name}_patch16_224"
    return model_name


# ============================================================================
#                           MAIN PROCESSING
# ============================================================================


def process_single_model(
    model_name: str,
    checkpoint_path: str,
    dataset: str,
    dataset_root: str,
    drop_path: float,
    output_dir: str,
    n_classes: int,
    n_samples_per_class: int,
    input_size: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    images: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    selected_classes: Optional[List[int]] = None,
    clip_quantile: Optional[float] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[int]]]:
    """
    Process a single model: load, run inference, and save PCA plots.

    Returns the images, labels, and selected_classes for reuse with other models.
    """
    print(f"\n▶ Processing {model_name} (drop_path={drop_path})")
    print(f"  Checkpoint: {checkpoint_path}")

    # Determine number of classes for model
    if dataset == "imagenet":
        num_model_classes = 1000
    elif dataset == "cifar100":
        num_model_classes = 100
    else:  # cifar10
        num_model_classes = 10

    # Get timm model name
    timm_model_name = get_timm_model_name(model_name, dataset)

    # Create and load model
    try:
        model = create_model(
            timm_model_name,
            pretrained=False,
            img_size=input_size,
            num_classes=num_model_classes
        )
    except Exception as e:
        print(f"  ⚠ Failed to create model {timm_model_name}: {e}")
        return images, labels, selected_classes

    try:
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in weights:
            state_dict = weights["state_dict"]
        elif "model" in weights:
            state_dict = weights["model"]
        else:
            state_dict = weights

        # Handle torch.compile() prefix: remove "_orig_mod." from keys if present
        state_dict = strip_compile_prefix(state_dict)

        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"  ⚠ Failed to load checkpoint: {e}")
        return images, labels, selected_classes

    model.eval().to(device)

    # Load data if not already loaded
    if images is None or labels is None:
        tfm = get_transform(input_size)

        if dataset == "imagenet":
            ds = ImageFolder(os.path.join(dataset_root, "val"), tfm)
        elif dataset == "cifar10":
            ds = CIFAR10(dataset_root, train=False, transform=tfm, download=False)
        else:  # cifar100
            ds = CIFAR100(dataset_root, train=False, transform=tfm, download=False)

        all_classes = list(set(extract_labels(ds)))

        # For CIFAR10, use all 10 classes; for others, sample
        if dataset == "cifar10":
            selected_classes = list(range(10))
        elif dataset == "imagenet":
            selected_classes = [0, 329, 658]
        else:
            random.seed(seed)
            selected_classes = random.sample(all_classes, min(n_classes, len(all_classes)))

        print(f"  Selected classes: {selected_classes}")

        images, labels = build_balanced_batch(
            ds,
            selected_classes=selected_classes,
            n_samples_per_class=n_samples_per_class,
            seed=seed,
            batch_size=batch_size,
            device=device,
        )

    images = images.to(device)
    labels = labels.to(device)

    # Register hooks and run inference
    pca_hook = PCAHook()
    pca_hook.register(model)

    with torch.no_grad():
        model(images)

    pca_hook.remove()

    # Create output directory
    model_output_dir = Path(output_dir) / dataset / model_name / f"drop_path={drop_path}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots for each layer
    for layer, projections in pca_hook.projection_per_layer.items():
        title = f"{simplify_model_name(model_name)}, {simplify_layer_name(layer)}"
        out_path = model_output_dir / f"{layer}.pdf"

        # Show legend only for first few P-head variants
        # show_legend = "one_P" in model_name or n_classes <= 10
        show_legend = False

        plot_sequences_by_label(
            projections, labels,
            title=title,
            out_path=out_path,
            legend=show_legend,
            close=True,
            clip_quantile=clip_quantile,
        )

    print(f"  ✓ Saved plots to {model_output_dir}")

    # Clean up model
    del model
    torch.cuda.empty_cache()

    return images, labels, selected_classes


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine which datasets to process
    datasets_to_process = args.datasets

    for dataset in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset.upper()}")
        print(f"{'='*60}")

        # Parse JSON and get model configurations
        if dataset == "imagenet":
            json_path = args.imagenet_json
            model_configs = parse_imagenet_json(json_path)
            dataset_root = args.imagenet_root
            input_size = 224
            n_classes = args.n_classes_imagenet
        elif dataset == "cifar10":
            json_path = args.cifar10_json
            model_configs = parse_cifar_json(json_path, dataset)
            dataset_root = args.cifar10_root
            input_size = 32
            n_classes = 10  # Use all classes for CIFAR10
        else:  # cifar100
            json_path = args.cifar100_json
            model_configs = parse_cifar_json(json_path, dataset)
            dataset_root = args.cifar100_root
            input_size = 32
            n_classes = args.n_classes_cifar100

        print(f"Found {len(model_configs)} model configurations in {json_path}")

        # Group by model+drop_path to avoid redundant data loading
        # Images/labels can be reused across different models if using same dataset
        images, labels, selected_classes = None, None, None

        for i, config in enumerate(model_configs):
            model_name = config['model_name']
            drop_path = config['drop_path']
            best_seed = config['best_seed']

            print(f"\n[{i+1}/{len(model_configs)}] {model_name}, dp={drop_path}, seed={best_seed}")

            # Get checkpoint path
            if dataset == "imagenet":
                ckpt_path = get_imagenet_checkpoint_path(
                    model_name, drop_path, best_seed
                )
            else:
                ckpt_path = get_cifar_checkpoint_path(
                    model_name, drop_path, best_seed, dataset
                )

            if ckpt_path is None:
                print(f"  ⚠ Checkpoint not found, skipping")
                continue

            images, labels, selected_classes = process_single_model(
                model_name=model_name,
                checkpoint_path=ckpt_path,
                dataset=dataset,
                dataset_root=dataset_root,
                drop_path=drop_path,
                output_dir=args.output_dir,
                n_classes=n_classes,
                n_samples_per_class=args.n_samples_per_class,
                input_size=input_size,
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
                images=images,
                labels=labels,
                selected_classes=selected_classes,
                clip_quantile=0.9 if args.clip_outliers else None,
            )

    print(f"\n{'='*60}")
    print("Done! All PCA plots saved.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA visualization for timm models across datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["imagenet", "cifar10", "cifar100"],
        default=["imagenet", "cifar10", "cifar100"],
        help="Datasets to process"
    )

    # JSON file paths
    parser.add_argument(
        "--imagenet-json",
        default=DEFAULT_IMAGENET_JSON,
        help="Path to ImageNet results JSON"
    )
    parser.add_argument(
        "--cifar10-json",
        default=DEFAULT_CIFAR10_JSON,
        help="Path to CIFAR10 results JSON"
    )
    parser.add_argument(
        "--cifar100-json",
        default=DEFAULT_CIFAR100_JSON,
        help="Path to CIFAR100 results JSON"
    )

    # Dataset root paths
    parser.add_argument(
        "--imagenet-root",
        default=DEFAULT_IMAGENET_ROOT,
        help="Root directory for ImageNet dataset"
    )
    parser.add_argument(
        "--cifar10-root",
        default=DEFAULT_CIFAR10_ROOT,
        help="Root directory for CIFAR10 dataset"
    )
    parser.add_argument(
        "--cifar100-root",
        default=DEFAULT_CIFAR100_ROOT,
        help="Root directory for CIFAR100 dataset"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for PCA plots"
    )

    # Sampling settings
    parser.add_argument(
        "--n-classes-imagenet",
        type=int,
        default=3,
        help="Number of classes to sample for ImageNet"
    )
    parser.add_argument(
        "--n-classes-cifar100",
        type=int,
        default=3,
        help="Number of classes to sample for CIFAR100"
    )
    parser.add_argument(
        "--n-samples-per-class",
        type=int,
        default=50,
        help="Number of samples per class"
    )

    # Other settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1002,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--clip-outliers",
        action="store_true",
        help="Constrain axis limits to show only the bottom 90%% quantile by magnitude"
    )

    args = parser.parse_args()
    main(args)
