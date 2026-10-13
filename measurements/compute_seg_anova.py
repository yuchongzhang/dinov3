"""Driver: ANOVA on DINOv3 last-layer tokens with ADE20K segmentation as
the per-image clustering.

Stages, mirroring ``measurements.compute_nested_anova``:
  1. Forward pass + segmentation-derived per-image clusters. Produces a
     :class:`LocalClusterRecords` table; optionally cached to disk.
     With ``--pca-components``, a preceding pass fits PCA on the same
     non-ignored tokens and the records are built from projected tokens.
  2. (Only with ``--global-clustering``.) Fit weighted MiniBatchKMeans
     on a subsample of local centroids.
  3. Compute the ANOVA decomposition. Two-level if global clustering is
     off; nested three-level if on.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from dinov3.data.datasets.ade20k import ADE20K, _Split as ADE20KSplit
from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
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
    compute_nested_anova,
    concat_records,
    fit_global_centers,
    l2_normalize_tokens,
)
from measurements.seg_anova import (
    TokenPCAAccumulator,
    TokenPCAProjection,
    cluster_image_tokens_from_segmentation,
    compute_two_level_anova,
    pool_segmentation_to_patches,
    project_tokens_onto_pca,
)


logger = logging.getLogger("measurements.compute_seg_anova")

DEFAULT_ADE20K_ROOT = "/home/yucz/projects/aip-papyan/shared/ADE20K/ADEChallengeData2016"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmentation-driven ANOVA on DINOv3 features (ADE20K)."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ade20k-root", default=DEFAULT_ADE20K_ROOT)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--crop-size", type=int, default=448,
                        help="Image and mask are square-resized to this size.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=20)

    cls_group = parser.add_mutually_exclusive_group()
    cls_group.add_argument("--exclude-cls", dest="exclude_cls", action="store_true")
    cls_group.add_argument("--include-cls", dest="exclude_cls", action="store_false")
    parser.set_defaults(exclude_cls=True)

    parser.add_argument("--ignore-label", type=int, default=0,
                        help="Patches with this label are dropped. ADE20K uses 0 for background.")

    global_group = parser.add_mutually_exclusive_group()
    global_group.add_argument("--global-clustering", dest="global_clustering",
                              action="store_true",
                              help="Run weighted MiniBatchKMeans across local centroids (default).")
    global_group.add_argument("--no-global-clustering", dest="global_clustering",
                              action="store_false",
                              help="Skip global clustering; report 2-level ANOVA only.")
    parser.set_defaults(global_clustering=True)

    parser.add_argument("--k-global", type=int, default=1000)
    parser.add_argument("--global-fit-sample-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--records-cache", type=Path, default=None,
                        help="If set, cache LocalClusterRecords here (skip Stage 1 if file exists).")
    parser.add_argument("--max-images", "--n-images", dest="max_images",
                        type=int, default=None)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "If set, fit PCA on all non-ignored token embeddings being decomposed "
            "and project tokens to the top K components before computing variance."
        ),
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


class SegEvalTransform:
    """Synchronized eval transform for an (image_PIL, mask_PIL) pair.

    Both inputs are square-resized to ``crop_size``: image with bilinear,
    mask with nearest. The image is converted to a normalized float tensor;
    the mask is returned as an int64 tensor of shape ``(crop_size, crop_size)``.
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size
        self._mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)

    def __call__(self, image, mask):
        s = self.crop_size
        image = TF.resize(image, [s, s], interpolation=InterpolationMode.BILINEAR)
        img_t = TF.to_tensor(image)                                       # (3, S, S) in [0, 1]
        img_t = (img_t - self._mean) / self._std

        mask = TF.resize(mask, [s, s], interpolation=InterpolationMode.NEAREST)
        mask_arr = np.array(mask, copy=False)
        mask_t = torch.as_tensor(mask_arr, dtype=torch.int64)            # (S, S)
        return img_t, mask_t


def default_output_path(checkpoint_path: Path, model_name: str, *,
                        global_clustering: bool, exclude_cls: bool,
                        pca_components: int | None = None) -> Path:
    step = infer_checkpoint_step(checkpoint_path)
    step_suffix = f"_training_{step}" if step is not None else ""
    mode_suffix = "_seg3" if global_clustering else "_seg2"
    cls_suffix = "_no_cls" if exclude_cls else ""
    pca_suffix = f"_pca{pca_components}" if pca_components is not None else ""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return Path("measurements") / "seg_anova_results" / f"{safe}{step_suffix}{cls_suffix}{pca_suffix}{mode_suffix}.json"


def derive_patch_size(token_count: int, crop_size: int) -> tuple[int, int]:
    """Return ``(patch_size, grid_edge)`` for a square patch-token grid."""

    grid_edge = int(round(token_count ** 0.5))
    if grid_edge * grid_edge != token_count:
        raise RuntimeError(
            f"Token count {token_count} is not a perfect square; cannot align to a patch grid."
        )
    if crop_size % grid_edge != 0:
        raise RuntimeError(
            f"crop_size {crop_size} not divisible by patch grid edge {grid_edge}."
        )
    return crop_size // grid_edge, grid_edge


def pca_projection_payload(projection: TokenPCAProjection | None) -> dict[str, Any]:
    if projection is None:
        return {"enabled": False}
    explained = projection.explained_variance.tolist()
    total_variance = projection.total_variance
    ratios = (
        [float(v / total_variance) for v in explained]
        if total_variance > 0
        else [0.0 for _ in explained]
    )
    return {
        "enabled": True,
        "num_components": projection.num_components,
        "input_dim": projection.input_dim,
        "output_dim": projection.num_components,
        "fit_num_tokens": projection.num_tokens,
        "fit_tokens": "non_ignored_segmentation_tokens",
        "centering": "pca_token_mean_after_existing_token_preprocessing",
        "explained_variance": explained,
        "explained_variance_ratio": ratios,
        "retained_variance": projection.retained_variance,
        "retained_variance_ratio": projection.retained_variance_ratio,
        "total_variance": total_variance,
    }


@torch.inference_mode()
def stage_fit_token_pca(
    *,
    model,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    exclude_cls: bool,
    crop_size: int,
    ignore_label: int,
    max_images: int | None,
    top_k: int,
    center_mean: torch.Tensor | None = None,
) -> TokenPCAProjection:
    mean_dev = (
        center_mean.to(device=device, dtype=torch.float32)
        if center_mean is not None
        else None
    )
    accumulator = TokenPCAAccumulator()
    cumulative = 0
    patch_size: int | None = None

    for batch_idx, (samples, masks) in enumerate(data_loader):
        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("pca-fit pass: batch %d / %d", batch_idx, len(data_loader))
        samples = samples.to(device, non_blocking=True)

        tokens = extract_last_layer_tokens(model, samples, autocast_dtype, exclude_cls=exclude_cls)
        if mean_dev is not None:
            tokens = tokens.to(torch.float32) - mean_dev
        tokens = l2_normalize_tokens(tokens)                                 # (B, T, D)
        B, T, _D = tokens.shape

        if patch_size is None:
            patch_size, grid_edge = derive_patch_size(T, crop_size)
            logger.info("derived patch_size=%d (token grid %dx%d)", patch_size, grid_edge, grid_edge)

        pooled = pool_segmentation_to_patches(
            masks, patch_size=patch_size, ignore_label=ignore_label
        )                                                                   # (B, T)
        accumulator.update(tokens, mask=(pooled != ignore_label))

        cumulative += B
        if max_images is not None and cumulative >= max_images:
            logger.info("pca-fit pass: hit --max-images=%d after batch %d, stopping early", max_images, batch_idx)
            break

    projection = accumulator.compute(top_k)
    logger.info(
        "fit PCA projection: components=%d, input_dim=%d, tokens=%d, retained_variance_ratio=%.4f",
        projection.num_components,
        projection.input_dim,
        projection.num_tokens,
        projection.retained_variance_ratio,
    )
    return projection


@torch.inference_mode()
def stage_extract_and_cluster(
    *,
    model,
    data_loader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    log_every: int,
    exclude_cls: bool,
    crop_size: int,
    ignore_label: int,
    max_images: int | None,
    center_mean: torch.Tensor | None = None,
    pca_projection: TokenPCAProjection | None = None,
) -> LocalClusterRecords:
    mean_dev = (
        center_mean.to(device=device, dtype=torch.float32)
        if center_mean is not None
        else None
    )
    parts: list[LocalClusterRecords] = []
    cumulative = 0
    patch_size: int | None = None

    for batch_idx, (samples, masks) in enumerate(data_loader):
        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("seg-cluster pass: batch %d / %d", batch_idx, len(data_loader))
        samples = samples.to(device, non_blocking=True)

        tokens = extract_last_layer_tokens(model, samples, autocast_dtype, exclude_cls=exclude_cls)
        if mean_dev is not None:
            tokens = tokens.to(torch.float32) - mean_dev
        tokens = l2_normalize_tokens(tokens)                                 # (B, T, D)
        B, T, _D = tokens.shape

        if patch_size is None:
            patch_size, grid_edge = derive_patch_size(T, crop_size)
            logger.info("derived patch_size=%d (token grid %dx%d)", patch_size, grid_edge, grid_edge)

        pooled = pool_segmentation_to_patches(
            masks, patch_size=patch_size, ignore_label=ignore_label
        )                                                                   # (B, T)

        if pca_projection is not None:
            tokens = project_tokens_onto_pca(tokens, pca_projection)

        ids = torch.arange(cumulative, cumulative + B, dtype=torch.int64)
        cumulative += B

        recs = cluster_image_tokens_from_segmentation(
            tokens, pooled, image_ids=ids, ignore_label=ignore_label
        )
        parts.append(recs)

        if max_images is not None and cumulative >= max_images:
            logger.info("hit --max-images=%d after batch %d, stopping early", max_images, batch_idx)
            break

    return concat_records(parts)


def build_payload(
    *,
    result,                                                                 # NestedAnovaResult or TwoLevelAnovaResult
    records: LocalClusterRecords,
    model_name: str,
    checkpoint_path: Path,
    config_file: Path,
    dataset_label: str,
    exclude_cls: bool,
    crop_size: int,
    patch_size: int,
    ignore_label: int,
    global_clustering: bool,
    k_global: int,
    global_fit_sample_size: int,
    seed: int,
    num_images: int,
    center_tokens: bool,
    token_mean_norm: float | None,
    pca_projection: TokenPCAProjection | None,
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

    payload: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": step,
        "config_file": str(config_file),
        "dataset": dataset_label,
        "representation": representation,
        "exclude_cls": exclude_cls,
        "l2_normalized": True,
        "center_tokens": center_tokens,
        "token_mean_norm": token_mean_norm,
        "pca_projection": pca_projection_payload(pca_projection),
        "mode": "three_level" if global_clustering else "two_level",
        "crop_size": crop_size,
        "patch_size": patch_size,
        "ignore_label": ignore_label,
        "num_images": num_images,
        "tokens_per_image": records.tokens_per_image,
        "num_local_clusters": result.num_local_clusters,
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
    if global_clustering:
        payload["num_global_clusters"] = result.num_global_clusters
        payload["global_clustering"] = {
            "algorithm": "weighted_minibatch_kmeans_subsample_then_assign",
            "k_global": k_global,
            "fit_sample_size": global_fit_sample_size,
            "seed": seed,
        }
    return payload


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.pca_components is not None and args.pca_components < 1:
        raise ValueError(f"--pca-components must be >= 1, got {args.pca_components}.")

    checkpoint_path = args.checkpoint.resolve()
    config_file = (args.config_file or infer_config_file(checkpoint_path)).resolve()
    config = load_config(config_file)
    model_name = infer_model_name(config, config_file)
    output_path = (
        args.output or default_output_path(
            checkpoint_path, model_name,
            global_clustering=args.global_clustering,
            exclude_cls=args.exclude_cls,
            pca_components=args.pca_components,
        )
    ).resolve()

    dataset_label = f"ade20k_{args.split}"
    logger.info("model_name=%s, dataset=%s", model_name, dataset_label)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("crop_size=%d, ignore_label=%d, exclude_cls=%s",
                args.crop_size, args.ignore_label, args.exclude_cls)
    logger.info("global_clustering=%s, k_global=%s",
                args.global_clustering, args.k_global if args.global_clustering else "n/a")

    logger.info("center_tokens=%s", args.center_tokens)
    logger.info("pca_components=%s", args.pca_components if args.pca_components is not None else "disabled")

    cache = args.records_cache.resolve() if args.records_cache else None
    records: LocalClusterRecords | None = None
    patch_size: int | None = None
    token_mean: torch.Tensor | None = None
    pca_projection: TokenPCAProjection | None = None

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
        cached_pca_components = blob.get("pca_num_components", None)
        if cached_pca_components is not None:
            cached_pca_components = int(cached_pca_components)
        if cached_pca_components != args.pca_components:
            raise RuntimeError(
                f"--records-cache {cache} was built with pca_components={cached_pca_components}, "
                f"but --pca-components={args.pca_components} was requested. Delete the cache "
                f"or pass the matching flag."
            )
        if cached_pca_components is not None:
            pca_projection = TokenPCAProjection(
                mean=blob["pca_mean"],
                components=blob["pca_basis"],
                explained_variance=blob["pca_explained_variance"],
                total_variance=float(blob["pca_total_variance"]),
                num_tokens=int(blob["pca_fit_num_tokens"]),
            )
            if records.embedding_dim != pca_projection.num_components:
                raise RuntimeError(
                    "Cached record dimension does not match cached PCA components: "
                    f"records D={records.embedding_dim}, PCA K={pca_projection.num_components}."
                )
        patch_size = int(blob.get("patch_size", args.crop_size // int(round(records.tokens_per_image ** 0.5))))
        logger.info(
            "loaded %d records (%d tokens), patch_size=%d, center_tokens=%s, pca_components=%s",
            records.num_records,
            records.num_tokens,
            patch_size,
            cached_centered,
            cached_pca_components if cached_pca_components is not None else "disabled",
        )

    if records is None:
        device = torch.device(args.device)
        model = build_model_for_eval(config, checkpoint_path)
        model.to(device).eval()
        autocast_dtype = get_autocast_dtype(config)

        split_enum = ADE20KSplit.VAL if args.split == "val" else ADE20KSplit.TRAIN
        transform = SegEvalTransform(crop_size=args.crop_size)
        dataset = ADE20K(split=split_enum, root=args.ade20k_root, transforms=transform)
        logger.info("dataset size = %d", len(dataset))

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
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

        if args.pca_components is not None:
            pca_projection = stage_fit_token_pca(
                model=model,
                data_loader=data_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                log_every=args.log_every,
                exclude_cls=args.exclude_cls,
                crop_size=args.crop_size,
                ignore_label=args.ignore_label,
                max_images=args.max_images,
                top_k=args.pca_components,
                center_mean=token_mean,
            )

        records = stage_extract_and_cluster(
            model=model,
            data_loader=data_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            log_every=args.log_every,
            exclude_cls=args.exclude_cls,
            crop_size=args.crop_size,
            ignore_label=args.ignore_label,
            max_images=args.max_images,
            center_mean=token_mean,
            pca_projection=pca_projection,
        )
        Hp = int(round(records.tokens_per_image ** 0.5))
        patch_size = args.crop_size // Hp

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache_blob = {
                "image_ids": records.image_ids,
                "sizes": records.sizes,
                "centroids": records.centroids,
                "within_ss": records.within_ss,
                "tokens_per_image": records.tokens_per_image,
                "patch_size": patch_size,
                "center_tokens": args.center_tokens,
                "token_mean": token_mean if token_mean is not None else torch.empty(0),
                "pca_num_components": args.pca_components,
            }
            if pca_projection is not None:
                cache_blob.update(
                    {
                        "pca_mean": pca_projection.mean,
                        "pca_basis": pca_projection.components,
                        "pca_explained_variance": pca_projection.explained_variance,
                        "pca_total_variance": pca_projection.total_variance,
                        "pca_fit_num_tokens": pca_projection.num_tokens,
                    }
                )
            torch.save(cache_blob, cache)
            logger.info("wrote records cache to %s", cache)

    num_images = int(records.image_ids.unique().numel())
    logger.info(
        "records=%d, tokens=%d, images=%d, mean clusters/image=%.2f",
        records.num_records, records.num_tokens, num_images,
        records.num_records / max(num_images, 1),
    )

    if args.global_clustering:
        centers = fit_global_centers(
            records,
            k_global=args.k_global,
            sample_size=args.global_fit_sample_size,
            seed=args.seed,
        )
        labels = assign_global(records, centers)
        result = compute_nested_anova(records, labels)
    else:
        result = compute_two_level_anova(records)

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
        dataset_label=dataset_label,
        exclude_cls=args.exclude_cls,
        crop_size=args.crop_size,
        patch_size=patch_size,
        ignore_label=args.ignore_label,
        global_clustering=args.global_clustering,
        k_global=args.k_global,
        global_fit_sample_size=args.global_fit_sample_size,
        seed=args.seed,
        num_images=num_images,
        center_tokens=args.center_tokens,
        token_mean_norm=token_mean_norm,
        pca_projection=pca_projection,
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
