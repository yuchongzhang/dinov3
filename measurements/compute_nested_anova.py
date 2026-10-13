"""Driver: compute nested ANOVA on DINOv3 last-layer tokens.

Three stages:
  1. Forward pass + per-image Ward agglomerative clustering of L2-normalized
     tokens. Produces a table of (image_id, n_l, centroid, within_ss).
     Optionally cached to disk for quick reruns.
  2. Fit weighted MiniBatchKMeans on a subsample of local centroids.
  3. Closed-form nested ANOVA on the records + global assignment.
"""

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
from measurements.compute_anova import (
    compute_dataset_token_mean,
    extract_last_layer_tokens,
    infer_checkpoint_step,
    infer_config_file,
    infer_model_name,
    load_config,
)
from measurements.nested_anova import (
    LocalClusterRecords,
    assign_global,
    cluster_image_tokens,
    compute_nested_anova,
    concat_records,
    fit_global_centers,
    l2_normalize_tokens,
)


logger = logging.getLogger("measurements.compute_nested_anova")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested token ANOVA on DINOv3 features (Ward local + weighted KMeans global)."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to teacher_checkpoint.pth.")
    parser.add_argument("--config-file", type=Path, default=None, help="Path to config.yaml (auto-inferred if omitted).")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset string. Defaults to train.dataset_path with TRAIN→VAL substitution; pass --use-train-dataset to keep TRAIN.",
    )
    parser.add_argument(
        "--use-train-dataset",
        action="store_true",
        help="Use train.dataset_path verbatim (no TRAIN→VAL substitution).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)
    parser.add_argument("--log-every", type=int, default=50)

    cls_group = parser.add_mutually_exclusive_group()
    cls_group.add_argument("--exclude-cls", dest="exclude_cls", action="store_true",
                           help="Drop the leading CLS token before clustering (default).")
    cls_group.add_argument("--include-cls", dest="exclude_cls", action="store_false",
                           help="Cluster CLS together with the patch tokens.")
    parser.set_defaults(exclude_cls=True)

    parser.add_argument(
        "--local-distance-threshold",
        type=float,
        default=0.5,
        help="Ward distance threshold (Euclidean on the unit sphere) for per-image clustering.",
    )
    parser.add_argument("--k-global", type=int, default=1000)
    parser.add_argument("--global-fit-sample-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--records-cache",
        type=Path,
        default=None,
        help="If set, cache LocalClusterRecords here (skip Stage 1 if file exists).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of images processed (for quick iteration).",
    )

    center_group = parser.add_mutually_exclusive_group()
    center_group.add_argument(
        "--center-tokens", dest="center_tokens", action="store_true",
        help="Subtract the dataset-level token mean before L2 normalization.",
    )
    center_group.add_argument(
        "--no-center-tokens", dest="center_tokens", action="store_false",
        help="Do not subtract any mean before L2 normalization (default).",
    )
    parser.set_defaults(center_tokens=False)
    return parser.parse_args()


def get_dataset_string(config: Any, override: str | None, *, use_train: bool) -> str:
    if override:
        return override
    cfg_dataset = OmegaConf.select(config, "train.dataset_path") or OmegaConf.select(config, "train.dataset")
    if not cfg_dataset:
        raise ValueError("Config has no train.dataset_path or train.dataset and no --dataset override was provided.")
    cfg_dataset = str(cfg_dataset)
    if use_train:
        return cfg_dataset
    # TRAIN → VAL substitution; idempotent if already VAL.
    swapped = re.sub(r"split=TRAIN\b", "split=VAL", cfg_dataset)
    if swapped == cfg_dataset and "split=" not in cfg_dataset:
        logger.warning("Could not infer val dataset from %r; using as-is.", cfg_dataset)
    return swapped


def default_output_path(checkpoint_path: Path, model_name: str, *, exclude_cls: bool) -> Path:
    step = infer_checkpoint_step(checkpoint_path)
    step_suffix = f"_training_{step}" if step is not None else ""
    token_suffix = "_no_cls" if exclude_cls else ""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return Path("measurements") / "nested_anova_results" / f"{safe}{step_suffix}{token_suffix}_nested_anova.json"


@torch.inference_mode()
def stage_extract_and_cluster(
    *,
    model,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    exclude_cls: bool,
    distance_threshold: float,
    max_images: int | None,
    center_mean: torch.Tensor | None = None,
) -> LocalClusterRecords:
    mean_dev = (
        center_mean.to(device=device, dtype=torch.float32)
        if center_mean is not None
        else None
    )
    parts: list[LocalClusterRecords] = []
    cumulative = 0
    for batch_idx, (samples, _targets) in enumerate(data_loader):
        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("local-cluster pass: batch %d / %d", batch_idx, len(data_loader))
        samples = samples.to(device, non_blocking=True)
        tokens = extract_last_layer_tokens(model, samples, autocast_dtype, exclude_cls=exclude_cls)
        if mean_dev is not None:
            tokens = tokens.to(torch.float32) - mean_dev
        tokens = l2_normalize_tokens(tokens)
        B = tokens.shape[0]
        ids = torch.arange(cumulative, cumulative + B, dtype=torch.int64)
        cumulative += B

        recs = cluster_image_tokens(
            tokens,
            distance_threshold=distance_threshold,
            image_ids=ids,
        )
        parts.append(recs)

        if max_images is not None and cumulative >= max_images:
            logger.info("hit --max-images=%d after batch %d, stopping early", max_images, batch_idx)
            break

    return concat_records(parts)


def build_payload(
    *,
    result,
    records: LocalClusterRecords,
    model_name: str,
    checkpoint_path: Path,
    config_file: Path,
    dataset: str,
    exclude_cls: bool,
    distance_threshold: float,
    k_global: int,
    global_fit_sample_size: int,
    seed: int,
    num_images: int,
    center_tokens: bool,
    token_mean_norm: float | None,
) -> dict[str, Any]:
    step = infer_checkpoint_step(checkpoint_path)
    representation = (
        "normalized_last_layer_patch_tokens"
        if exclude_cls
        else "normalized_last_layer_cls_plus_patch_tokens"
    )
    sizes = records.sizes.to(dtype=torch.float64)
    clusters_per_image: dict[int, int] = {}
    for img_id in records.image_ids.tolist():
        clusters_per_image[img_id] = clusters_per_image.get(img_id, 0) + 1
    cluster_counts = torch.tensor(list(clusters_per_image.values()), dtype=torch.float64)
    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": step,
        "config_file": str(config_file),
        "dataset": dataset,
        "representation": representation,
        "exclude_cls": exclude_cls,
        "l2_normalized": True,
        "center_tokens": center_tokens,
        "token_mean_norm": token_mean_norm,
        "local_clustering": {
            "linkage": "ward",
            "metric": "euclidean",
            "distance_threshold": distance_threshold,
        },
        "global_clustering": {
            "algorithm": "weighted_minibatch_kmeans_subsample_then_assign",
            "k_global": k_global,
            "fit_sample_size": global_fit_sample_size,
            "seed": seed,
        },
        "num_images": num_images,
        "tokens_per_image": records.tokens_per_image,
        "num_local_clusters": result.num_local_clusters,
        "num_global_clusters": result.num_global_clusters,
        "num_tokens": result.num_tokens,
        "mean_local_clusters_per_image": float(cluster_counts.mean().item()) if cluster_counts.numel() else 0.0,
        "min_local_clusters_per_image": int(cluster_counts.min().item()) if cluster_counts.numel() else 0,
        "max_local_clusters_per_image": int(cluster_counts.max().item()) if cluster_counts.numel() else 0,
        "mean_local_cluster_size": float(sizes.mean().item()) if sizes.numel() else 0.0,
        "variances": result.variances(),
        "percent_of_total": result.percentages(),
        "anova_error": result.anova_error,
        "absolute_anova_error": result.absolute_anova_error,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint_path = args.checkpoint.resolve()
    config_file = (args.config_file or infer_config_file(checkpoint_path)).resolve()
    config = load_config(config_file)
    dataset_string = get_dataset_string(config, args.dataset, use_train=args.use_train_dataset)
    model_name = infer_model_name(config, config_file)
    output_path = (
        args.output or default_output_path(checkpoint_path, model_name, exclude_cls=args.exclude_cls)
    ).resolve()

    logger.info("model_name=%s", model_name)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("config_file=%s", config_file)
    logger.info("dataset=%s", dataset_string)
    logger.info("exclude_cls=%s", args.exclude_cls)
    logger.info("local_distance_threshold=%s", args.local_distance_threshold)
    logger.info("k_global=%s, fit_sample_size=%s", args.k_global, args.global_fit_sample_size)
    logger.info("center_tokens=%s", args.center_tokens)

    cache = args.records_cache.resolve() if args.records_cache else None
    records: LocalClusterRecords | None = None
    token_mean: torch.Tensor | None = None
    if cache is not None and cache.exists():
        logger.info("loading cached records from %s", cache)
        blob = torch.load(cache, map_location="cpu", weights_only=True)
        records = LocalClusterRecords(
            image_ids=blob["image_ids"],
            sizes=blob["sizes"],
            centroids=blob["centroids"],
            within_ss=blob["within_ss"],
            tokens_per_image=int(blob["tokens_per_image"]),
        )
        cached_centered = bool(blob.get("center_tokens", False))
        if cached_centered != args.center_tokens:
            raise RuntimeError(
                f"--records-cache {cache} was built with center_tokens={cached_centered}, "
                f"but --center-tokens={args.center_tokens} was requested. Delete the cache "
                f"or pass the matching flag."
            )
        cached_mean = blob.get("token_mean", None)
        if cached_mean is not None and getattr(cached_mean, "numel", lambda: 0)() > 0:
            token_mean = cached_mean
        logger.info("loaded %d records (%d tokens), center_tokens=%s",
                    records.num_records, records.num_tokens, cached_centered)

    if records is None:
        device = torch.device(args.device)
        model = build_model_for_eval(config, checkpoint_path)
        model.to(device)
        model.eval()
        autocast_dtype = get_autocast_dtype(config)

        transform = make_classification_eval_transform(
            resize_size=args.resize_size, crop_size=args.crop_size
        )
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

        if args.center_tokens:
            token_mean = compute_dataset_token_mean(
                model=model,
                data_loader=data_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                log_every=args.log_every,
                exclude_cls=args.exclude_cls,
                max_images=args.max_images,
            )
            logger.info("computed token mean: ||mean||=%.4f", float(token_mean.norm().item()))

        records = stage_extract_and_cluster(
            model=model,
            data_loader=data_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            log_every=args.log_every,
            exclude_cls=args.exclude_cls,
            distance_threshold=args.local_distance_threshold,
            max_images=args.max_images,
            center_mean=token_mean,
        )

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "image_ids": records.image_ids,
                    "sizes": records.sizes,
                    "centroids": records.centroids,
                    "within_ss": records.within_ss,
                    "tokens_per_image": records.tokens_per_image,
                    "center_tokens": args.center_tokens,
                    "token_mean": token_mean if token_mean is not None else torch.empty(0),
                },
                cache,
            )
            logger.info("wrote records cache to %s", cache)

    num_images = int(records.image_ids.unique().numel())
    logger.info(
        "records=%d, tokens=%d, images=%d, mean clusters/image=%.2f",
        records.num_records, records.num_tokens, num_images, records.num_records / max(num_images, 1),
    )

    centers = fit_global_centers(
        records,
        k_global=args.k_global,
        sample_size=args.global_fit_sample_size,
        seed=args.seed,
    )
    labels = assign_global(records, centers)
    result = compute_nested_anova(records, labels)

    token_mean_norm = (
        float(token_mean.norm().item())
        if token_mean is not None and token_mean.numel() > 0
        else None
    )
    payload = build_payload(
        result=result,
        records=records,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        dataset=dataset_string,
        exclude_cls=args.exclude_cls,
        distance_threshold=args.local_distance_threshold,
        k_global=args.k_global,
        global_fit_sample_size=args.global_fit_sample_size,
        seed=args.seed,
        num_images=num_images,
        center_tokens=args.center_tokens,
        token_mean_norm=token_mean_norm,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("wrote %s", output_path)
    logger.info("anova_error: %.3e (|error|=%.3e)", result.anova_error, result.absolute_anova_error)
    logger.info("variances: %s", result.variances())
    logger.info("percentages: %s", result.percentages())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
