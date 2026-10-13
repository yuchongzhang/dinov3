from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dinov3.data.transforms import CROP_DEFAULT_SIZE, RESIZE_DEFAULT_SIZE
from measurements.dino_measurement_utils import (
    build_eval_dataset,
    extract_block_tokens,
    get_dataset_string,
    infer_checkpoint_step,
    infer_config_file,
    infer_model_name,
    load_config,
    load_model_for_measurement,
    make_simple_data_loader,
    safe_output_stem,
)


logger = logging.getLogger("measurements.compute_block_cossim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DINOv3 block-wise token cosine similarity.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to teacher_checkpoint.pth.")
    parser.add_argument("--config-file", type=Path, default=None, help="Path to config.yaml.")
    parser.add_argument("--dataset", default=None, help="DINO dataset string. Defaults to train.dataset_path.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-mean-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def compute_cosine_similarity(x: torch.Tensor) -> float:
    """Average off-diagonal pair-wise cosine similarity for tokens shaped (B, N, C)."""

    if x.ndim != 3:
        raise ValueError(f"Expected tensor with shape (B, N, C), got {tuple(x.shape)}")
    x = x.float()
    x_norm = F.normalize(x, p=2, dim=-1)
    sim = torch.bmm(x_norm, x_norm.transpose(1, 2))

    token_count = sim.size(1)
    if token_count < 2:
        raise ValueError("Cosine similarity requires at least two tokens.")
    off_diag_sum = sim.sum(dim=(1, 2)) - torch.diagonal(sim, dim1=1, dim2=2).sum(dim=1)
    cos_sim = off_diag_sum / (token_count * (token_count - 1))
    return float(cos_sim.mean().item())


class GlobalMeanAccumulator:
    def __init__(self) -> None:
        self.sums: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.counts: "OrderedDict[str, int]" = OrderedDict()

    def update(self, layer_tokens: "OrderedDict[str, torch.Tensor]") -> None:
        for layer_name, tokens in layer_tokens.items():
            if tokens.ndim != 3:
                raise ValueError(f"Expected {layer_name} tokens with shape (B, T, C), got {tuple(tokens.shape)}")
            token_sum = tokens.detach().float().sum(dim=(0, 1)).cpu()
            token_count = tokens.shape[0] * tokens.shape[1]
            if layer_name not in self.sums:
                self.sums[layer_name] = token_sum
                self.counts[layer_name] = token_count
            else:
                self.sums[layer_name] += token_sum
                self.counts[layer_name] += token_count

    def finalize(self) -> "OrderedDict[str, torch.Tensor]":
        if not self.sums:
            raise RuntimeError("Cannot finalize global means before seeing any batches.")
        return OrderedDict((layer_name, self.sums[layer_name] / self.counts[layer_name]) for layer_name in self.sums)


def default_output_path(checkpoint_path: Path, model_name: str) -> Path:
    return Path("measurements") / "block_cossim_results" / f"{safe_output_stem(model_name, checkpoint_path)}_block_cossim.json"


def iter_image_batches(data_loader, device: torch.device, *, log_every: int, pass_name: str):
    for batch_idx, (images, _) in enumerate(data_loader):
        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("%s pass: batch %d / %d", pass_name, batch_idx, len(data_loader))
        yield images.to(device, non_blocking=True)


def compute_global_means(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_batches: int | None,
    log_every: int,
) -> tuple["OrderedDict[str, torch.Tensor]", int, int]:
    accumulator = GlobalMeanAccumulator()
    num_batches = 0
    num_images = 0

    for images in iter_image_batches(data_loader, device, log_every=log_every, pass_name="mean"):
        if max_batches is not None and num_batches >= max_batches:
            break
        layer_tokens = extract_block_tokens(model, images, autocast_dtype=autocast_dtype)
        accumulator.update(layer_tokens)
        num_batches += 1
        num_images += images.shape[0]

    return accumulator.finalize(), num_batches, num_images


def compute_eval_cossim(
    *,
    model: torch.nn.Module,
    eval_images: torch.Tensor,
    autocast_dtype: torch.dtype,
    global_means: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, float]":
    layer_tokens = extract_block_tokens(model, eval_images, autocast_dtype=autocast_dtype)
    results: "OrderedDict[str, float]" = OrderedDict()

    for layer_name, tokens in layer_tokens.items():
        if layer_name not in global_means:
            raise KeyError(f"Missing global mean for {layer_name}.")
        centered = tokens.float() - global_means[layer_name].to(device=tokens.device, dtype=torch.float32)
        results[layer_name] = compute_cosine_similarity(centered)

    return results


def build_payload(
    *,
    results: "OrderedDict[str, float]",
    model_name: str,
    checkpoint_path: Path,
    config_file: Path,
    dataset: str,
    batch_size: int,
    mean_batches: int,
    mean_images: int,
    eval_batch_size: int,
) -> dict[str, Any]:
    block_items = [(name, value) for name, value in results.items() if name.startswith("block_")]
    final_norm_cossim = results["norm_final"]
    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": infer_checkpoint_step(checkpoint_path),
        "config_file": str(config_file),
        "dataset": dataset,
        "batch_size": batch_size,
        "mean_batches": mean_batches,
        "mean_images": mean_images,
        "eval_batch_size": eval_batch_size,
        "num_blocks": len(block_items),
        "layer_names": list(results.keys()),
        "block_cossim": [value for _, value in block_items],
        "final_norm_cossim": final_norm_cossim,
        "all_cossim": list(results.values()),
        "layers": dict(results),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    checkpoint_path = args.checkpoint.resolve()
    config_file = (args.config_file or infer_config_file(checkpoint_path)).resolve()
    config = load_config(config_file)
    model_name = infer_model_name(config, config_file)
    dataset_string = get_dataset_string(config, args.dataset)
    output_path = (args.output or default_output_path(checkpoint_path, model_name)).resolve()

    logger.info("model_name=%s", model_name)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("config_file=%s", config_file)
    logger.info("dataset=%s", dataset_string)

    device = torch.device(args.device)
    dataset = build_eval_dataset(
        dataset_string=dataset_string,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
    )
    eval_loader = make_simple_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    mean_loader = make_simple_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed + 1,
        drop_last=False,
    )

    eval_images, _ = next(iter(eval_loader))
    eval_images = eval_images.to(device, non_blocking=True)

    model, autocast_dtype = load_model_for_measurement(config, checkpoint_path, device)
    global_means, mean_batches, mean_images = compute_global_means(
        model=model,
        data_loader=mean_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        max_batches=args.max_mean_batches,
        log_every=args.log_every,
    )
    results = compute_eval_cossim(
        model=model,
        eval_images=eval_images,
        autocast_dtype=autocast_dtype,
        global_means=global_means,
    )
    payload = build_payload(
        results=results,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        dataset=dataset_string,
        batch_size=args.batch_size,
        mean_batches=mean_batches,
        mean_images=mean_images,
        eval_batch_size=eval_images.shape[0],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

