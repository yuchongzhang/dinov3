from __future__ import annotations

import inspect
import logging
import re
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dinov3.configs.config import get_default_config
from dinov3.data import make_dataset
from dinov3.data.transforms import CROP_DEFAULT_SIZE, RESIZE_DEFAULT_SIZE, make_classification_eval_transform
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval


logger = logging.getLogger("measurements.dino_measurement_utils")


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


def load_config(config_file: Path):
    return OmegaConf.merge(get_default_config(), OmegaConf.load(config_file))


def infer_model_name(config: Any, config_file: Path) -> str:
    wandb_name = OmegaConf.select(config, "wandb.name")
    if wandb_name:
        return str(wandb_name)
    return config_file.parent.name


def get_dataset_string(config: Any, override: str | None) -> str:
    if override:
        return override
    dataset = OmegaConf.select(config, "train.dataset_path")
    if not dataset:
        dataset = OmegaConf.select(config, "train.dataset")
    if not dataset:
        raise ValueError("No dataset was provided and neither train.dataset_path nor train.dataset exists in config.")
    return str(dataset)


def safe_output_stem(model_name: str, checkpoint_path: Path) -> str:
    checkpoint_step = infer_checkpoint_step(checkpoint_path)
    step_suffix = f"_training_{checkpoint_step}" if checkpoint_step is not None else ""
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return f"{safe_model_name}{step_suffix}"


def check_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")


def autocast_context(device: torch.device, autocast_dtype: torch.dtype):
    if device.type == "cuda" and autocast_dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)
    return nullcontext()


def load_model_for_measurement(config: Any, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.dtype]:
    check_device(device)
    model = build_model_for_eval(config, checkpoint_path)
    model.to(device)
    model.eval()
    return model, get_autocast_dtype(config)


def build_eval_dataset(
    *,
    dataset_string: str,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
):
    transform = make_classification_eval_transform(resize_size=resize_size, crop_size=crop_size)
    return make_dataset(dataset_str=dataset_string, transform=transform)


def make_simple_data_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        generator=generator,
        **kwargs,
    )


def dataset_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "get_targets"):
        targets = dataset.get_targets()
        if targets is not None:
            return np.asarray(targets)
    if hasattr(dataset, "targets") and dataset.targets is not None:
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels") and dataset.labels is not None:
        return np.asarray(dataset.labels)
    if hasattr(dataset, "samples"):
        return np.asarray([target for _, target in dataset.samples])
    if hasattr(dataset, "_labels"):
        return np.asarray(dataset._labels)
    raise AttributeError("Could not find labels/targets on the dataset.")


def build_balanced_batch(
    dataset,
    *,
    selected_classes: Sequence[int],
    samples_per_class: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = dataset_targets(dataset)
    rng = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []

    for class_id in selected_classes:
        class_id = int(class_id)
        pool = np.flatnonzero(targets == class_id)
        if pool.size < samples_per_class:
            raise ValueError(f"Class {class_id} has only {pool.size} samples, need {samples_per_class}.")
        selected_indices.append(rng.choice(pool, size=samples_per_class, replace=False))

    indices = np.concatenate(selected_indices)
    rng.shuffle(indices)
    subset = Subset(dataset, indices.tolist())
    loader = make_simple_data_loader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    image_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for images, labels in loader:
        image_parts.append(images)
        label_parts.append(labels)

    images = torch.cat(image_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    if device is not None:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    return images, labels


def get_num_layers(model: torch.nn.Module) -> int:
    if hasattr(model, "n_blocks"):
        return int(model.n_blocks)
    if hasattr(model, "blocks"):
        return len(model.blocks)
    if hasattr(model, "downsample_layers"):
        return len(model.downsample_layers)
    raise AttributeError("Expected the model to expose n_blocks, blocks, or downsample_layers.")


def _patch_tokens_to_sequence(patch_tokens: torch.Tensor) -> torch.Tensor:
    if patch_tokens.ndim == 3:
        return patch_tokens
    if patch_tokens.ndim == 4:
        return patch_tokens.flatten(2).transpose(1, 2)
    raise ValueError(f"Expected patch tokens with shape (B, T, C) or (B, C, H, W), got {tuple(patch_tokens.shape)}")


def concatenate_layer_tokens(layer_output) -> torch.Tensor:
    """Return tokens in DINO order: CLS, storage/register tokens, patch tokens."""

    class_token = None
    extra_tokens = None
    patch_tokens = layer_output

    if isinstance(layer_output, (tuple, list)):
        if len(layer_output) == 3:
            patch_tokens, class_token, extra_tokens = layer_output
        elif len(layer_output) == 2:
            patch_tokens, class_token = layer_output
        elif len(layer_output) == 1:
            patch_tokens = layer_output[0]
        else:
            raise ValueError(f"Unsupported layer output with {len(layer_output)} entries.")

    sequence_parts: list[torch.Tensor] = []
    if class_token is not None:
        if class_token.ndim != 2:
            raise ValueError(f"Expected class token with shape (B, C), got {tuple(class_token.shape)}")
        sequence_parts.append(class_token.unsqueeze(1))
    if extra_tokens is not None and extra_tokens.numel() > 0:
        if extra_tokens.ndim != 3:
            raise ValueError(f"Expected extra tokens with shape (B, R, C), got {tuple(extra_tokens.shape)}")
        sequence_parts.append(extra_tokens)
    sequence_parts.append(_patch_tokens_to_sequence(patch_tokens))
    return torch.cat(sequence_parts, dim=1)


def _get_intermediate_outputs(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    layer_indices: Iterable[int],
    norm: bool,
):
    signature = inspect.signature(model.get_intermediate_layers)
    kwargs: dict[str, Any] = {
        "n": list(layer_indices),
        "return_class_token": True,
        "norm": norm,
    }
    if "return_extra_tokens" in signature.parameters:
        kwargs["return_extra_tokens"] = True
    return model.get_intermediate_layers(images, **kwargs)


def apply_final_norm(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    if getattr(model, "untie_cls_and_patch_norms", False):
        special_token_count = int(getattr(model, "n_storage_tokens", 0)) + 1
        cls_and_extra = model.cls_norm(tokens[:, :special_token_count])
        patch_tokens = model.norm(tokens[:, special_token_count:])
        return torch.cat((cls_and_extra, patch_tokens), dim=1)
    return model.norm(tokens)


@torch.inference_mode()
def extract_block_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    autocast_dtype: torch.dtype,
) -> "OrderedDict[str, torch.Tensor]":
    device = images.device
    n_layers = get_num_layers(model)
    with autocast_context(device, autocast_dtype):
        outputs = _get_intermediate_outputs(model, images, layer_indices=range(n_layers), norm=False)

        layer_tokens: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for idx, layer_output in enumerate(outputs):
            layer_tokens[f"block_{idx}"] = concatenate_layer_tokens(layer_output)
        if not layer_tokens:
            raise RuntimeError("No intermediate layer outputs were returned.")
        layer_tokens["norm_final"] = apply_final_norm(model, next(reversed(layer_tokens.values())))
    return layer_tokens
