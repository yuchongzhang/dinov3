"""Nested ANOVA with k-means at both clustering levels.

The decomposition is the same as :mod:`measurements.nested_anova`:

    SS_total = SS_within_local + SS_between_local_in_global + SS_between_global

Here local clusters are formed by running k-means separately inside each
image, and global clusters are formed by weighted k-means over the local
centroids. The ANOVA math itself is reused from ``nested_anova`` so the
variance identity follows the same size-weighted centroid convention.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans

from measurements.nested_anova import LocalClusterRecords, compute_nested_anova
from measurements.seg_anova import TokenPCAAccumulator, TokenPCAProjection, project_tokens_onto_pca


def _to_cpu_f64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float64)


def _validate_k(name: str, value: int, maximum: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    if value > maximum:
        raise ValueError(f"{name}={value} cannot exceed the number of points ({maximum}).")


def cluster_image_tokens_kmeans(
    tokens: torch.Tensor,
    *,
    k_local: int,
    image_ids: Sequence[int] | torch.Tensor | None = None,
    seed: int = 0,
    kmeans_kwargs: dict | None = None,
) -> LocalClusterRecords:
    """Run k-means within each image and return local-cluster records.

    Parameters
    ----------
    tokens : (B, T, D) tensor
        Tokens after any CLS filtering, centering, normalization, and optional
        PCA projection.
    k_local : int
        Number of k-means clusters to fit per image.
    image_ids : optional length-B sequence
        Stable image identifiers. Defaults to ``range(B)``.
    seed : int
        Base random seed. Image ``b`` uses ``seed + b``.
    """

    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, D), got {tuple(tokens.shape)}")
    B, T, D = tokens.shape
    _validate_k("k_local", k_local, T)

    tokens_cpu = _to_cpu_f64(tokens)
    if image_ids is None:
        image_ids_t = torch.arange(B, dtype=torch.int64)
    else:
        image_ids_t = torch.as_tensor(image_ids, dtype=torch.int64)
        if image_ids_t.shape[0] != B:
            raise ValueError(f"image_ids length {image_ids_t.shape[0]} != batch size {B}")

    out_image_ids: list[int] = []
    out_sizes: list[int] = []
    out_centroids: list[torch.Tensor] = []
    out_within: list[float] = []

    kwargs = dict(n_clusters=k_local, random_state=seed, n_init=10)
    if kmeans_kwargs:
        kwargs.update(kmeans_kwargs)

    for b in range(B):
        x = tokens_cpu[b]  # (T, D)
        kwargs["random_state"] = seed + b
        labels = KMeans(**kwargs).fit_predict(x.numpy())

        for label in np.unique(labels):
            mask = labels == label
            members = x[torch.from_numpy(mask)]
            centroid = members.mean(dim=0)
            within = (members - centroid.unsqueeze(0)).pow(2).sum().item()
            out_image_ids.append(int(image_ids_t[b].item()))
            out_sizes.append(int(mask.sum()))
            out_centroids.append(centroid)
            out_within.append(float(within))

    return LocalClusterRecords(
        image_ids=torch.tensor(out_image_ids, dtype=torch.int64),
        sizes=torch.tensor(out_sizes, dtype=torch.int64),
        centroids=torch.stack(out_centroids, dim=0).to(dtype=torch.float64),
        within_ss=torch.tensor(out_within, dtype=torch.float64),
        tokens_per_image=T,
    )


def fit_global_kmeans(
    records: LocalClusterRecords,
    *,
    k_global: int,
    seed: int = 0,
    kmeans_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run weighted k-means on local centroids.

    Returns ``(centers, labels)``. The labels assign each row in ``records`` to
    a global cluster. Local cluster sizes are used as sample weights. If
    ``k_global`` is larger than the number of local records, it is capped at
    ``records.num_records`` so small ``--max-images`` runs stay convenient.
    """

    if records.num_records == 0:
        raise ValueError("Cannot fit global k-means on an empty record table.")
    if k_global < 1:
        raise ValueError(f"k_global must be >= 1, got {k_global}.")
    n_clusters = min(k_global, records.num_records)

    kwargs = dict(n_clusters=n_clusters, random_state=seed, n_init=10)
    if kmeans_kwargs:
        kwargs.update(kmeans_kwargs)

    centroids = records.centroids.numpy()
    weights = records.sizes.to(dtype=torch.float64).numpy()
    model = KMeans(**kwargs)
    model.fit(centroids, sample_weight=weights)

    centers = torch.from_numpy(model.cluster_centers_).to(dtype=torch.float64)
    labels = torch.from_numpy(model.labels_).to(dtype=torch.int64)
    return centers, labels


def compute_k_means_anova(
    records: LocalClusterRecords,
    labels: torch.Tensor,
):
    """Compute the nested ANOVA result for k-means record assignments."""

    return compute_nested_anova(records, labels)


__all__ = [
    "LocalClusterRecords",
    "TokenPCAAccumulator",
    "TokenPCAProjection",
    "cluster_image_tokens_kmeans",
    "compute_k_means_anova",
    "fit_global_kmeans",
    "project_tokens_onto_pca",
]
