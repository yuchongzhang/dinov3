#!/usr/bin/env python3
"""
Compute block-wise cosine similarity for transformer models.

Computes the average pair-wise cosine similarity of token embeddings at the output
of each transformer block and saves results to JSON for later analysis/plotting.

Usage examples:

1. Arbitrary models via CLI:
   python compute_block_cossim.py \
       --dataset imagenet --dataset-path /path/to/imagenet \
       --models model1:checkpoint1.pth model2:checkpoint2.pth \
       --output results.json

2. Same model with different drop paths (convenience mode):
   python compute_block_cossim.py \
       --dataset imagenet --dataset-path /path/to/imagenet \
       --model-name deit3_base_0P_qk_norm \
       --artifacts-dir /path/to/artifacts \
       --drop-paths 0.1 0.3 0.4 \
       --seed 2030 \
       --output results.json

3. Using a config file:
   python compute_block_cossim.py \
       --dataset imagenet --dataset-path /path/to/imagenet \
       --config models_config.json \
       --output results.json
"""

import argparse
import collections
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


# -------------------------------------------------------------------- #
#                            Math helpers                               #
# -------------------------------------------------------------------- #

def compute_cosine_similarity(x: torch.Tensor) -> float:
    """Average pair-wise cosine similarity for a batch of sequences.

    Args:
        x: Tensor of shape (B, N, C) where B is batch size, N is token
           count, and C is embedding dimension.

    Returns:
        Scalar mean CosSim over the batch.
    """
    x_norm = F.normalize(x, p=2, dim=-1)  # (B, N, C)
    sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, N, N)

    n = sim.size(1)
    off_diag_sum = sim.sum(dim=(1, 2)) - torch.diagonal(sim, dim1=1, dim2=2).sum(dim=1)
    cos_sim = off_diag_sum / (n * (n - 1))
    return cos_sim.mean().item()


# -------------------------------------------------------------------- #
#                           Forward hooks                               #
# -------------------------------------------------------------------- #

class GlobalMeanHook:
    """Accumulates global mean of activations across the dataset."""

    def __init__(self):
        self.sums = collections.defaultdict(lambda: 0)
        self.counts = collections.defaultdict(int)

    def _output_hook(self, layer_name):
        def fn(_m, _inp, _out):
            x = _out
            if x.dim() == 3:  # (B, N, C)
                B, N, C = x.shape
                self.sums[f"{layer_name}_out"] += x.sum(dim=(0, 1))
                self.counts[f"{layer_name}_out"] += B * N
            else:  # (B, C)
                self.sums[f"{layer_name}_out"] += x.sum(dim=0)
                self.counts[f"{layer_name}_out"] += x.size(0)
        return fn

    def register(self, model):
        self.handles = []
        for idx, blk in enumerate(model.blocks):
            self.handles.append(blk.register_forward_hook(self._output_hook(f"block_{idx}")))
        self.handles.append(model.norm.register_forward_hook(self._output_hook("norm_final")))

    def remove(self):
        for h in self.handles:
            h.remove()

    def means(self, device):
        return {k: (self.sums[k] / self.counts[k]).to(device) for k in self.sums}


class TokenCosSimHook:
    """Collects cosine similarity at the output of each transformer block."""

    def __init__(self):
        self.block_cs: List[float] = []
        self.norm_out_cs: List[float] = []

    @staticmethod
    def _output_hook(store: List[float], global_mean):
        def fn(_module, _inp, _out):
            x = _out - global_mean.to(_out.device)
            store.append(compute_cosine_similarity(x))
        return fn

    def register(self, model, global_mean_dict):
        self._handles = []
        for idx, blk in enumerate(model.blocks):
            self._handles.append(
                blk.register_forward_hook(
                    self._output_hook(self.block_cs, global_mean_dict[f"block_{idx}_out"])
                )
            )
        self._handles.append(
            model.norm.register_forward_hook(
                self._output_hook(self.norm_out_cs, global_mean_dict["norm_final_out"])
            )
        )

    def remove(self):
        for h in self._handles:
            h.remove()

    def get_results(self) -> List[float]:
        """Returns block-wise cossim followed by final norm output."""
        return self.block_cs + self.norm_out_cs


# -------------------------------------------------------------------- #
#                           Data loading                                #
# -------------------------------------------------------------------- #

def build_dataloader(ds_name: str, root: str, size: int, bs: int) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if ds_name == "imagenet":
        ds = ImageFolder(os.path.join(root, "val"), tfm)
    elif ds_name == "cifar10":
        ds = CIFAR10(root, train=False, transform=tfm, download=False)
    elif ds_name == "cifar100":
        ds = CIFAR100(root, train=False, transform=tfm, download=False)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    return DataLoader(ds, batch_size=bs, shuffle=True)


# -------------------------------------------------------------------- #
#                        Model configuration                            #
# -------------------------------------------------------------------- #

def resolve_checkpoint_path(
    artifacts_dir: str,
    model_name: str,
    drop_path: float,
    seed: int
) -> str:
    """Construct checkpoint path from artifacts directory structure."""
    base_name = f"{model_name}_patch16_224"

    if drop_path == 0.4:
        run_name = f"{base_name}_in1k_seed={seed}"
    else:
        run_name = f"{base_name}_in1k_seed={seed}_drop_path={drop_path}"

    return os.path.join(artifacts_dir, base_name, run_name, "model_best.pth.tar")


def build_model_configs_from_drop_paths(
    model_name: str,
    artifacts_dir: str,
    drop_paths: List[float],
    seeds: List[int]
) -> List[Dict]:
    """Build model configurations for same model with different drop paths."""
    configs = []
    for dp, seed in zip(drop_paths, seeds):
        checkpoint = resolve_checkpoint_path(artifacts_dir, model_name, dp, seed)
        configs.append({
            "model_name": f"{model_name}_patch16_224",
            "checkpoint": checkpoint,
            "label": f"{model_name}_drop_path={dp}",
            "drop_path": dp,
            "seed": seed
        })
    return configs


def build_model_configs_from_cli(model_specs: List[str]) -> List[Dict]:
    """Parse model:checkpoint pairs from CLI."""
    configs = []
    for spec in model_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid model spec '{spec}'. Expected format: model_name:checkpoint_path")
        model_name, checkpoint = spec.split(":", 1)
        configs.append({
            "model_name": model_name,
            "checkpoint": checkpoint,
            "label": model_name
        })
    return configs


def load_model_configs_from_file(config_path: str) -> List[Dict]:
    """Load model configurations from JSON file.

    Expected format:
    {
        "models": [
            {"model_name": "...", "checkpoint": "...", "label": "..."},
            ...
        ]
    }
    """
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data["models"]


# -------------------------------------------------------------------- #
#                         Core computation                              #
# -------------------------------------------------------------------- #

def compute_cossim_for_model(
    model_name: str,
    checkpoint: str,
    dataloader: DataLoader,
    eval_batch: torch.Tensor,
    device: str,
    input_size: int = 224
) -> List[float]:
    """Compute block-wise cosine similarity for a single model."""
    # Create and load model
    model = create_model(model_name, pretrained=False, img_size=input_size, num_classes=1000)

    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in weights:
        model.load_state_dict(weights["state_dict"])
    elif "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)

    model.eval().to(device)

    # Compute global means across dataset
    global_mean_hook = GlobalMeanHook()
    global_mean_hook.register(model)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            model(images)
        global_means = global_mean_hook.means(device)
        global_mean_hook.remove()

    # Compute cosine similarity on eval batch
    hooker = TokenCosSimHook()
    hooker.register(model, global_means)

    with torch.no_grad():
        model(eval_batch)

    hooker.remove()

    return hooker.get_results()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build model configurations based on input mode
    if args.config:
        model_configs = load_model_configs_from_file(args.config)
    elif args.models:
        model_configs = build_model_configs_from_cli(args.models)
    elif args.model_name and args.artifacts_dir and args.drop_paths:
        # Expand single seed to all drop paths for backward compatibility
        seeds = args.seed if len(args.seed) > 1 else args.seed * len(args.drop_paths)
        model_configs = build_model_configs_from_drop_paths(
            args.model_name,
            args.artifacts_dir,
            args.drop_paths,
            seeds
        )
    else:
        raise ValueError(
            "Must provide one of: --config, --models, or (--model-name + --artifacts-dir + --drop-paths)"
        )

    # Validate checkpoints exist
    for cfg in model_configs:
        if not os.path.exists(cfg["checkpoint"]):
            print(f"Warning: Checkpoint not found: {cfg['checkpoint']}")

    # Build dataloader and get eval batch
    loader = build_dataloader(args.dataset, args.dataset_path, args.input_size, args.batch_size)
    eval_images, _ = next(iter(loader))
    eval_images = eval_images.to(device)

    # Compute cosine similarity for each model
    results = {
        "metadata": {
            "dataset": args.dataset,
            "input_size": args.input_size,
            "batch_size": args.batch_size
        },
        "models": {}
    }

    for cfg in model_configs:
        label = cfg["label"]
        print(f"Processing: {label}")

        if not os.path.exists(cfg["checkpoint"]):
            print(f"  Skipping (checkpoint not found)")
            continue

        # Need fresh dataloader for global mean computation
        loader_for_mean = build_dataloader(
            args.dataset, args.dataset_path, args.input_size, args.batch_size
        )

        cossim_values = compute_cossim_for_model(
            cfg["model_name"],
            cfg["checkpoint"],
            loader_for_mean,
            eval_images,
            device,
            args.input_size
        )

        n_blocks = len(cossim_values) - 1  # last one is final norm

        results["models"][label] = {
            "model_name": cfg["model_name"],
            "checkpoint": cfg["checkpoint"],
            "num_blocks": n_blocks,
            "block_cossim": cossim_values[:-1],
            "final_norm_cossim": cossim_values[-1],
            "all_cossim": cossim_values
        }

        # Add optional metadata
        if "drop_path" in cfg:
            results["models"][label]["drop_path"] = cfg["drop_path"]
        if "seed" in cfg:
            results["models"][label]["seed"] = cfg["seed"]

        print(f"  Computed {n_blocks} blocks + final norm")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute block-wise cosine similarity for transformer models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        choices=["imagenet", "cifar10", "cifar100"],
        required=True,
        help="Dataset to use for evaluation"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Root directory of the dataset"
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="block_cossim.json",
        help="Output JSON file path"
    )

    # Model specification options (mutually exclusive modes)
    mode_group = parser.add_argument_group("Model specification (use one mode)")

    mode_group.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file with model specifications"
    )
    mode_group.add_argument(
        "--models",
        nargs="+",
        help="Model specifications as model_name:checkpoint_path pairs"
    )

    # Drop path comparison mode
    drop_path_group = parser.add_argument_group("Drop path comparison mode")
    drop_path_group.add_argument(
        "--model-name",
        type=str,
        help="Base model name (without _patch16_224 suffix)"
    )
    drop_path_group.add_argument(
        "--artifacts-dir",
        type=str,
        help="Path to training artifacts directory"
    )
    drop_path_group.add_argument(
        "--drop-paths",
        type=float,
        nargs="+",
        help="List of drop path rates to compare"
    )
    drop_path_group.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[2030],
        help="Random seed(s) for checkpoint selection. Provide one per drop path, or single seed for all (default: 2030)"
    )

    args = parser.parse_args()
    main(args)
