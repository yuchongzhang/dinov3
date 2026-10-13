"""Driver: nested ANOVA with k-means local and global clustering."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

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
from measurements.k_means_anova import (
    LocalClusterRecords,
    TokenPCAAccumulator,
    TokenPCAProjection,
    cluster_image_tokens_kmeans,
    compute_k_means_anova,
    fit_global_kmeans,
    project_tokens_onto_pca,
)
from measurements.nested_anova import concat_records, l2_normalize_tokens


logger = logging.getLogger("measurements.compute_k_means_anova")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested token ANOVA on DINOv3 features using k-means for local and global clustering."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to teacher_checkpoint.pth.")
    parser.add_argument("--config-file", type=Path, default=None, help="Path to config.yaml (auto-inferred if omitted).")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset string. Defaults to train.dataset_path with TRAIN->VAL substitution.",
    )
    parser.add_argument(
        "--use-train-dataset",
        action="store_true",
        help="Use train.dataset_path verbatim.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)
    parser.add_argument("--log-every", type=int, default=50)

    cls_group = parser.add_mutually_exclusive_group()
    cls_group.add_argument("--exclude-cls", dest="exclude_cls", action="store_true")
    cls_group.add_argument("--include-cls", dest="exclude_cls", action="store_false")
    parser.set_defaults(exclude_cls=True)

    parser.add_argument("--k-local", type=int, default=8, help="Number of k-means clusters per image.")
    parser.add_argument("--k-global", type=int, default=1000, help="Number of global k-means clusters.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None, help="Optional image cap for quick runs.")
    parser.add_argument(
        "--records-cache",
        type=Path,
        default=None,
        help="If set, cache local k-means records here and reuse them on matching reruns.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Project tokens onto the top K PCA components before per-image k-means.",
    )

    center_group = parser.add_mutually_exclusive_group()
    center_group.add_argument(
        "--center-tokens",
        dest="center_tokens",
        action="store_true",
        help="Subtract the dataset-level token mean before L2 normalization.",
    )
    center_group.add_argument(
        "--no-center-tokens",
        dest="center_tokens",
        action="store_false",
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
    swapped = re.sub(r"split=TRAIN\b", "split=VAL", cfg_dataset)
    if swapped == cfg_dataset and "split=" not in cfg_dataset:
        logger.warning("Could not infer val dataset from %r; using as-is.", cfg_dataset)
    return swapped


def default_output_path(
    checkpoint_path: Path,
    model_name: str,
    *,
    exclude_cls: bool,
    k_local: int,
    k_global: int,
    pca_components: int | None,
) -> Path:
    step = infer_checkpoint_step(checkpoint_path)
    step_suffix = f"_training_{step}" if step is not None else ""
    cls_suffix = "_no_cls" if exclude_cls else ""
    pca_suffix = f"_pca{pca_components}" if pca_components is not None else ""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    name = f"{safe}{step_suffix}{cls_suffix}_kmeans_l{k_local}_g{k_global}{pca_suffix}.json"
    return Path("measurements") / "k_means_anova_results" / name


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
        "fit_tokens": "all_processed_tokens",
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

    for batch_idx, (samples, _targets) in enumerate(data_loader):
        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("pca-fit pass: batch %d / %d", batch_idx, len(data_loader))
        samples = samples.to(device, non_blocking=True)
        tokens = extract_last_layer_tokens(model, samples, autocast_dtype, exclude_cls=exclude_cls)
        if mean_dev is not None:
            tokens = tokens.to(torch.float32) - mean_dev
        tokens = l2_normalize_tokens(tokens)
        accumulator.update(tokens)

        cumulative += tokens.shape[0]
        if max_images is not None and cumulative >= max_images:
            logger.info("pca-fit pass: hit --max-images=%d after batch %d, stopping", max_images, batch_idx)
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
    k_local: int,
    seed: int,
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

    for batch_idx, (samples, _targets) in enumerate(data_loader):
        if log_every > 0 and batch_idx % log_every == 0:
            logger.info("local k-means pass: batch %d / %d", batch_idx, len(data_loader))
        samples = samples.to(device, non_blocking=True)
        tokens = extract_last_layer_tokens(model, samples, autocast_dtype, exclude_cls=exclude_cls)
        if mean_dev is not None:
            tokens = tokens.to(torch.float32) - mean_dev
        tokens = l2_normalize_tokens(tokens)
        if pca_projection is not None:
            tokens = project_tokens_onto_pca(tokens, pca_projection)

        B = tokens.shape[0]
        ids = torch.arange(cumulative, cumulative + B, dtype=torch.int64)
        recs = cluster_image_tokens_kmeans(
            tokens,
            k_local=k_local,
            image_ids=ids,
            seed=seed + cumulative,
        )
        parts.append(recs)
        cumulative += B

        if max_images is not None and cumulative >= max_images:
            logger.info("hit --max-images=%d after batch %d, stopping", max_images, batch_idx)
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
    k_local: int,
    k_global: int,
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
        "pca_projection": pca_projection_payload(pca_projection),
        "local_clustering": {
            "algorithm": "kmeans",
            "k_local": k_local,
            "seed": seed,
        },
        "global_clustering": {
            "algorithm": "weighted_kmeans",
            "k_global": k_global,
            "seed": seed,
        },
        "num_images": num_images,
        "tokens_per_image": records.tokens_per_image,
        "embedding_dim": records.embedding_dim,
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


def load_cached_records(
    cache: Path,
    *,
    center_tokens: bool,
    k_local: int,
    pca_components: int | None,
) -> tuple[LocalClusterRecords, torch.Tensor | None, TokenPCAProjection | None]:
    blob = torch.load(cache, map_location="cpu", weights_only=True)
    records = LocalClusterRecords(
        image_ids=blob["image_ids"],
        sizes=blob["sizes"],
        centroids=blob["centroids"],
        within_ss=blob["within_ss"],
        tokens_per_image=int(blob["tokens_per_image"]),
    )

    cached_centered = bool(blob.get("center_tokens", False))
    cached_k_local = int(blob.get("k_local", -1))
    cached_pca_components = blob.get("pca_num_components", None)
    if cached_pca_components is not None:
        cached_pca_components = int(cached_pca_components)

    if cached_centered != center_tokens:
        raise RuntimeError(
            f"--records-cache {cache} was built with center_tokens={cached_centered}, "
            f"but center_tokens={center_tokens} was requested."
        )
    if cached_k_local != k_local:
        raise RuntimeError(
            f"--records-cache {cache} was built with k_local={cached_k_local}, "
            f"but k_local={k_local} was requested."
        )
    if cached_pca_components != pca_components:
        raise RuntimeError(
            f"--records-cache {cache} was built with pca_components={cached_pca_components}, "
            f"but pca_components={pca_components} was requested."
        )

    token_mean = blob.get("token_mean", None)
    if token_mean is not None and getattr(token_mean, "numel", lambda: 0)() == 0:
        token_mean = None

    pca_projection = None
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

    return records, token_mean, pca_projection


def save_cached_records(
    cache: Path,
    *,
    records: LocalClusterRecords,
    center_tokens: bool,
    k_local: int,
    token_mean: torch.Tensor | None,
    pca_projection: TokenPCAProjection | None,
) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "image_ids": records.image_ids,
        "sizes": records.sizes,
        "centroids": records.centroids,
        "within_ss": records.within_ss,
        "tokens_per_image": records.tokens_per_image,
        "center_tokens": center_tokens,
        "k_local": k_local,
        "token_mean": token_mean if token_mean is not None else torch.empty(0),
        "pca_num_components": pca_projection.num_components if pca_projection is not None else None,
    }
    if pca_projection is not None:
        blob.update(
            {
                "pca_mean": pca_projection.mean,
                "pca_basis": pca_projection.components,
                "pca_explained_variance": pca_projection.explained_variance,
                "pca_total_variance": pca_projection.total_variance,
                "pca_fit_num_tokens": pca_projection.num_tokens,
            }
        )
    torch.save(blob, cache)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.k_local < 1:
        raise ValueError(f"--k-local must be >= 1, got {args.k_local}.")
    if args.k_global < 1:
        raise ValueError(f"--k-global must be >= 1, got {args.k_global}.")
    if args.pca_components is not None and args.pca_components < 1:
        raise ValueError(f"--pca-components must be >= 1, got {args.pca_components}.")

    checkpoint_path = args.checkpoint.resolve()
    config_file = (args.config_file or infer_config_file(checkpoint_path)).resolve()
    config = load_config(config_file)
    dataset_string = get_dataset_string(config, args.dataset, use_train=args.use_train_dataset)
    model_name = infer_model_name(config, config_file)
    output_path = (
        args.output or default_output_path(
            checkpoint_path,
            model_name,
            exclude_cls=args.exclude_cls,
            k_local=args.k_local,
            k_global=args.k_global,
            pca_components=args.pca_components,
        )
    ).resolve()

    logger.info("model_name=%s", model_name)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("config_file=%s", config_file)
    logger.info("dataset=%s", dataset_string)
    logger.info("exclude_cls=%s, k_local=%d, k_global=%d", args.exclude_cls, args.k_local, args.k_global)
    logger.info("center_tokens=%s, pca_components=%s", args.center_tokens, args.pca_components or "disabled")

    cache = args.records_cache.resolve() if args.records_cache else None
    records: LocalClusterRecords | None = None
    token_mean: torch.Tensor | None = None
    pca_projection: TokenPCAProjection | None = None

    if cache is not None and cache.exists():
        logger.info("loading cached records from %s", cache)
        records, token_mean, pca_projection = load_cached_records(
            cache,
            center_tokens=args.center_tokens,
            k_local=args.k_local,
            pca_components=args.pca_components,
        )
        logger.info("loaded %d records (%d tokens)", records.num_records, records.num_tokens)

    if records is None:
        device = torch.device(args.device)
        model = build_model_for_eval(config, checkpoint_path)
        model.to(device).eval()
        autocast_dtype = get_autocast_dtype(config)

        transform = make_classification_eval_transform(
            resize_size=args.resize_size,
            crop_size=args.crop_size,
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

        if args.pca_components is not None:
            pca_projection = stage_fit_token_pca(
                model=model,
                data_loader=data_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                log_every=args.log_every,
                exclude_cls=args.exclude_cls,
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
            k_local=args.k_local,
            seed=args.seed,
            max_images=args.max_images,
            center_mean=token_mean,
            pca_projection=pca_projection,
        )

        if cache is not None:
            save_cached_records(
                cache,
                records=records,
                center_tokens=args.center_tokens,
                k_local=args.k_local,
                token_mean=token_mean,
                pca_projection=pca_projection,
            )
            logger.info("wrote records cache to %s", cache)

    num_images = int(records.image_ids.unique().numel())
    logger.info(
        "records=%d, tokens=%d, images=%d, mean clusters/image=%.2f",
        records.num_records,
        records.num_tokens,
        num_images,
        records.num_records / max(num_images, 1),
    )

    centers, labels = fit_global_kmeans(records, k_global=args.k_global, seed=args.seed)
    if centers.shape[0] < args.k_global:
        logger.info("capped global k-means clusters at %d local records", centers.shape[0])
    result = compute_k_means_anova(records, labels)

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
        k_local=args.k_local,
        k_global=args.k_global,
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
