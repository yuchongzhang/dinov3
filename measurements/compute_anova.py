from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from dinov3.configs.config import get_default_config
from dinov3.data import make_data_loader, make_dataset
from dinov3.data.transforms import CROP_DEFAULT_SIZE, RESIZE_DEFAULT_SIZE, make_classification_eval_transform
from dinov3.eval.setup import get_autocast_dtype
from dinov3.models import build_model_for_eval
from measurements.anova import (
    TokenAnovaMeans,
    TokenMeanAccumulator,
    TokenVarianceAccumulator,
    exclude_cls_token,
)


logger = logging.getLogger("measurements.compute_anova")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute weighted token ANOVA for a DINO teacher checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to teacher_checkpoint.pth.")
    parser.add_argument("--config-file", type=Path, default=None, help="Path to the model config.yaml.")
    parser.add_argument("--dataset", default=None, help="Dataset string. Defaults to train.dataset_path from config.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for model inference. DINO checkpoint loading expects CUDA.",
    )
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--exclude-cls",
        action="store_true",
        help="Exclude the leading CLS token from the token ANOVA and use patch tokens only.",
    )
    return parser.parse_args()


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


def get_dataset_string(config: Any, override: str | None) -> str:
    if override:
        return override
    dataset = OmegaConf.select(config, "train.dataset_path")
    if not dataset:
        dataset = OmegaConf.select(config, "train.dataset")
    if not dataset:
        raise ValueError("No dataset was provided and neither train.dataset_path nor train.dataset exists in config.")
    return str(dataset)


def default_output_path(checkpoint_path: Path, model_name: str, *, exclude_cls: bool = False) -> Path:
    checkpoint_step = infer_checkpoint_step(checkpoint_path)
    step_suffix = f"_training_{checkpoint_step}" if checkpoint_step is not None else ""
    token_suffix = "_no_cls" if exclude_cls else ""
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return Path("measurements") / "anova_results" / f"{safe_model_name}{step_suffix}{token_suffix}_anova.json"


@torch.inference_mode()
def extract_last_layer_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
    autocast_dtype: torch.dtype,
    *,
    exclude_cls: bool = False,
) -> torch.Tensor:
    device_type = images.device.type
    autocast_enabled = device_type == "cuda" and autocast_dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=device_type, enabled=autocast_enabled, dtype=autocast_dtype):
        outputs = model.get_intermediate_layers(images, n=1, return_class_token=True, norm=True)
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one intermediate layer, got {len(outputs)}")
    patch_tokens, class_tokens = outputs[0]
    tokens = torch.cat((class_tokens.unsqueeze(1), patch_tokens), dim=1)
    if exclude_cls:
        tokens = exclude_cls_token(tokens)
    return tokens


def iter_batches(data_loader, device: torch.device, log_every: int, pass_name: str):
    for batch_idx, (samples, targets) in enumerate(data_loader):
        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("%s pass: batch %d / %d", pass_name, batch_idx, len(data_loader))
        yield samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)


@torch.inference_mode()
def compute_dataset_token_mean(
    *,
    model,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    exclude_cls: bool,
    max_images: int | None = None,
) -> torch.Tensor:
    """Single forward pass returning the (D,) float64 mean of last-layer tokens
    *before* any L2 normalization. The mean is taken over every patch token
    (and CLS, if ``exclude_cls=False``) across the dataset.

    Honors ``max_images`` so the mean comes from the same image subset that
    downstream stages use.
    """

    sum_t: torch.Tensor | None = None
    count = 0
    cumulative = 0
    for batch_idx, batch in enumerate(data_loader):
        samples = batch[0]
        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("token-mean pass: batch %d / %d", batch_idx, len(data_loader))
        samples = samples.to(device, non_blocking=True)
        tokens = extract_last_layer_tokens(model, samples, autocast_dtype, exclude_cls=exclude_cls)
        flat = tokens.reshape(-1, tokens.shape[-1]).to(dtype=torch.float64).cpu()
        if sum_t is None:
            sum_t = flat.sum(dim=0)
        else:
            sum_t += flat.sum(dim=0)
        count += flat.shape[0]
        cumulative += tokens.shape[0]
        if max_images is not None and cumulative >= max_images:
            logger.info("token-mean pass: hit --max-images=%d after batch %d, stopping",
                        max_images, batch_idx)
            break
    if sum_t is None or count == 0:
        raise RuntimeError("compute_dataset_token_mean: no tokens were observed.")
    return sum_t / count


def run_mean_pass(
    model,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    *,
    exclude_cls: bool = False,
) -> TokenAnovaMeans:
    accumulator = TokenMeanAccumulator()
    for samples, targets in iter_batches(data_loader, device, log_every, "mean"):
        tokens = extract_last_layer_tokens(model, samples, autocast_dtype, exclude_cls=exclude_cls)
        accumulator.update(tokens, targets)
    return accumulator.finalize()


def run_variance_pass(
    model,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    means: TokenAnovaMeans,
    log_every: int,
    *,
    exclude_cls: bool = False,
):
    accumulator = TokenVarianceAccumulator(means)
    for samples, targets in iter_batches(data_loader, device, log_every, "variance"):
        tokens = extract_last_layer_tokens(model, samples, autocast_dtype, exclude_cls=exclude_cls)
        accumulator.update(tokens, targets)
    return accumulator.finalize()


def build_payload(
    *,
    result,
    means: TokenAnovaMeans,
    model_name: str,
    checkpoint_path: Path,
    config_file: Path,
    dataset: str,
    exclude_cls: bool,
) -> dict[str, Any]:
    checkpoint_step = infer_checkpoint_step(checkpoint_path)
    representation = (
        "normalized_last_layer_patch_tokens"
        if exclude_cls
        else "normalized_last_layer_cls_plus_patch_tokens"
    )
    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "config_file": str(config_file),
        "dataset": dataset,
        "representation": representation,
        "exclude_cls": exclude_cls,
        "num_samples": means.num_sequences,
        "num_classes": means.num_classes,
        "tokens_per_sample": means.tokens_per_sequence,
        "variances": result.variances(),
        "percent_of_total": result.percentages(),
        "anova_error": result.anova_error,
        "absolute_anova_error": result.absolute_anova_error,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. DINO checkpoint loading in this repo expects CUDA.")

    checkpoint_path = args.checkpoint.resolve()
    config_file = (args.config_file or infer_config_file(checkpoint_path)).resolve()
    config = load_config(config_file)
    dataset_string = get_dataset_string(config, args.dataset)
    model_name = infer_model_name(config, config_file)
    output_path = (
        args.output or default_output_path(checkpoint_path, model_name, exclude_cls=args.exclude_cls)
    ).resolve()

    logger.info("model_name=%s", model_name)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("config_file=%s", config_file)
    logger.info("dataset=%s", dataset_string)
    logger.info("exclude_cls=%s", args.exclude_cls)

    device = torch.device(args.device)
    model = build_model_for_eval(config, checkpoint_path)
    model.to(device)
    model.eval()
    autocast_dtype = get_autocast_dtype(config)

    transform = make_classification_eval_transform(resize_size=args.resize_size, crop_size=args.crop_size)
    dataset = make_dataset(dataset_str=dataset_string, transform=transform)
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler_type=None,
        shuffle=False,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    means = run_mean_pass(
        model,
        data_loader,
        device,
        autocast_dtype,
        args.log_every,
        exclude_cls=args.exclude_cls,
    )
    result = run_variance_pass(
        model,
        data_loader,
        device,
        autocast_dtype,
        means,
        args.log_every,
        exclude_cls=args.exclude_cls,
    )
    payload = build_payload(
        result=result,
        means=means,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        dataset=dataset_string,
        exclude_cls=args.exclude_cls,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
