"""Nested ANOVA over DINOv3 token embeddings on the unit sphere.

Decomposes the empirical Euclidean SS of L2-normalized last-layer tokens
into three components keyed off a *data-driven* hierarchical grouping:

    SS_total = SS_within_local + SS_between_local_in_global + SS_between_global

where each token belongs to a "local cluster" (Ward agglomerative within
its image) and each local cluster belongs to a "global cluster"
(weighted MiniBatchKMeans across all local centroids of the dataset).

The size-weighted centroid convention is what makes the cross-terms
vanish and the identity hold exactly:

    mu_g = (sum_{l in g} n_l * centroid_l) / (sum_{l in g} n_l)
    mu_G = (sum_l n_l * centroid_l) / (sum_l n_l)

Companion to ``measurements.anova``; same dataclass / accumulator style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans


def l2_normalize_tokens(tokens: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Return tokens divided by their L2 norm along the feature dimension."""

    if tokens.ndim < 2:
        raise ValueError(f"Expected tokens with at least 2 dims, got shape {tuple(tokens.shape)}")
    norms = tokens.norm(dim=-1, keepdim=True).clamp_min(eps)
    return tokens / norms


@dataclass(frozen=True)
class LocalClusterRecords:
    """Sufficient statistics for the nested-ANOVA pipeline.

    All arrays are aligned along axis 0 (one row per local cluster).
    Stored on CPU in float64/int64 so the global-step math is exact.
    """

    image_ids: torch.Tensor       # int64 (M,)
    sizes: torch.Tensor           # int64 (M,)            n_l, in tokens
    centroids: torch.Tensor       # float64 (M, D)        mean of normalized tokens in cluster
    within_ss: torch.Tensor       # float64 (M,)          sum_t ||x_t - centroid||^2
    tokens_per_image: int

    @property
    def num_records(self) -> int:
        return int(self.sizes.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.sizes.sum().item())

    @property
    def embedding_dim(self) -> int:
        return int(self.centroids.shape[1])

    def append(self, other: "LocalClusterRecords") -> "LocalClusterRecords":
        if self.num_records == 0:
            return other
        if other.num_records == 0:
            return self
        if self.embedding_dim != other.embedding_dim:
            raise ValueError("Cannot append records with different embedding dims.")
        if self.tokens_per_image != other.tokens_per_image:
            raise ValueError(
                "Cannot append records produced from different sequence lengths "
                f"({self.tokens_per_image} vs {other.tokens_per_image})."
            )
        return LocalClusterRecords(
            image_ids=torch.cat([self.image_ids, other.image_ids]),
            sizes=torch.cat([self.sizes, other.sizes]),
            centroids=torch.cat([self.centroids, other.centroids], dim=0),
            within_ss=torch.cat([self.within_ss, other.within_ss]),
            tokens_per_image=self.tokens_per_image,
        )


@dataclass(frozen=True)
class NestedAnovaResult:
    """Empirical nested-ANOVA variances (each = SS_component / total_tokens)."""

    within_local: float
    between_local_in_global: float
    between_global: float
    total: float
    num_tokens: int
    num_local_clusters: int
    num_global_clusters: int

    @property
    def components_sum(self) -> float:
        return self.within_local + self.between_local_in_global + self.between_global

    @property
    def anova_error(self) -> float:
        return self.total - self.components_sum

    @property
    def absolute_anova_error(self) -> float:
        return abs(self.anova_error)

    def variances(self) -> dict[str, float]:
        return {
            "within_local": self.within_local,
            "between_local_in_global": self.between_local_in_global,
            "between_global": self.between_global,
            "total": self.total,
        }

    def percentages(self) -> dict[str, float]:
        if self.total == 0:
            return {k: 0.0 for k in ("within_local", "between_local_in_global", "between_global", "components_sum")}
        return {
            "within_local": 100.0 * self.within_local / self.total,
            "between_local_in_global": 100.0 * self.between_local_in_global / self.total,
            "between_global": 100.0 * self.between_global / self.total,
            "components_sum": 100.0 * self.components_sum / self.total,
        }


def _to_cpu_f64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float64)


def cluster_image_tokens(
    tokens: torch.Tensor,
    *,
    distance_threshold: float,
    image_ids: Sequence[int] | torch.Tensor | None = None,
    linkage: str = "ward",
) -> LocalClusterRecords:
    """Per-image Ward agglomerative clustering on already-L2-normalized tokens.

    Parameters
    ----------
    tokens : (B, T, D) tensor
        L2-normalized DINOv3 last-layer tokens. Caller is responsible for
        any CLS exclusion before this call.
    distance_threshold : float
        Ward distance threshold (Euclidean). On the unit sphere a cosine
        similarity ``c`` corresponds to Euclidean ``sqrt(2 (1 - c))``.
    image_ids : optional sequence of length B
        Stable identifiers, e.g. global indices in the dataset. Defaults
        to ``range(B)``.
    linkage : str
        Linkage to use. Default ``"ward"`` (only valid with Euclidean).

    Returns
    -------
    LocalClusterRecords
        One row per local cluster, all on CPU in float64 / int64.
    """

    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, D), got {tuple(tokens.shape)}")
    B, T, D = tokens.shape
    if T < 1:
        raise ValueError("Need at least one token per image.")

    tokens_cpu = _to_cpu_f64(tokens)

    if image_ids is None:
        image_ids = torch.arange(B, dtype=torch.int64)
    else:
        image_ids = torch.as_tensor(image_ids, dtype=torch.int64)
        if image_ids.shape[0] != B:
            raise ValueError(f"image_ids length {image_ids.shape[0]} != batch size {B}")

    out_image_ids: list[int] = []
    out_sizes: list[int] = []
    out_centroids: list[torch.Tensor] = []
    out_within: list[float] = []

    for b in range(B):
        x = tokens_cpu[b]  # (T, D)
        x_np = x.numpy()
        if T == 1:
            labels = np.zeros(1, dtype=np.int64)
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(x_np)

        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = labels == lbl
            n_l = int(mask.sum())
            members = x[torch.from_numpy(mask)]
            centroid = members.mean(dim=0)                                     # (D,)
            within = (members - centroid.unsqueeze(0)).pow(2).sum().item()
            out_image_ids.append(int(image_ids[b].item()))
            out_sizes.append(n_l)
            out_centroids.append(centroid)
            out_within.append(float(within))

    if not out_sizes:
        return LocalClusterRecords(
            image_ids=torch.empty(0, dtype=torch.int64),
            sizes=torch.empty(0, dtype=torch.int64),
            centroids=torch.empty(0, D, dtype=torch.float64),
            within_ss=torch.empty(0, dtype=torch.float64),
            tokens_per_image=T,
        )

    return LocalClusterRecords(
        image_ids=torch.tensor(out_image_ids, dtype=torch.int64),
        sizes=torch.tensor(out_sizes, dtype=torch.int64),
        centroids=torch.stack(out_centroids, dim=0).to(dtype=torch.float64),
        within_ss=torch.tensor(out_within, dtype=torch.float64),
        tokens_per_image=T,
    )


def concat_records(parts: Iterable[LocalClusterRecords]) -> LocalClusterRecords:
    parts = list(parts)
    if not parts:
        raise ValueError("concat_records called with no parts.")
    out = parts[0]
    for p in parts[1:]:
        out = out.append(p)
    return out


def fit_global_centers(
    records: LocalClusterRecords,
    *,
    k_global: int,
    sample_size: int | None = None,
    seed: int = 0,
    minibatch_kwargs: dict | None = None,
) -> torch.Tensor:
    """Fit MiniBatchKMeans centers on a (sample-weighted) subsample of local centroids.

    Returns
    -------
    centers : (k_global, D) float64 CPU tensor.
        These are the *fitted* k-means centers; ANOVA uses size-weighted
        per-global means re-derived in :func:`compute_nested_anova` so
        the variance identity holds exactly regardless of the optimizer's
        terminal accuracy.
    """

    if records.num_records == 0:
        raise ValueError("Cannot fit global centers on an empty record table.")
    if k_global < 1:
        raise ValueError(f"k_global must be >= 1, got {k_global}")

    n = records.num_records
    rng = np.random.default_rng(seed)
    if sample_size is None or sample_size >= n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)

    centroids = records.centroids[idx].numpy()
    weights = records.sizes[idx].to(dtype=torch.float64).numpy()

    kwargs = dict(
        n_clusters=min(k_global, len(idx)),
        random_state=seed,
        n_init="auto",
        max_iter=100,
        batch_size=min(4096, len(idx)),
    )
    if minibatch_kwargs:
        kwargs.update(minibatch_kwargs)
    model = MiniBatchKMeans(**kwargs)
    model.fit(centroids, sample_weight=weights)

    return torch.from_numpy(model.cluster_centers_).to(dtype=torch.float64)


def assign_global(records: LocalClusterRecords, centers: torch.Tensor) -> torch.Tensor:
    """Assign each local centroid to its nearest fitted center."""

    if records.num_records == 0:
        return torch.empty(0, dtype=torch.int64)
    if centers.ndim != 2 or centers.shape[1] != records.embedding_dim:
        raise ValueError(
            f"Centers shape {tuple(centers.shape)} incompatible with D={records.embedding_dim}"
        )
    centers64 = centers.to(dtype=torch.float64, device="cpu")
    cents = records.centroids                              # (M, D)
    # ||c - g||^2 = ||c||^2 - 2 c . g + ||g||^2; argmin over g
    cn = (cents * cents).sum(dim=1, keepdim=True)          # (M, 1)
    gn = (centers64 * centers64).sum(dim=1, keepdim=True).T  # (1, K)
    cross = cents @ centers64.T                            # (M, K)
    d2 = cn + gn - 2 * cross
    return d2.argmin(dim=1).to(dtype=torch.int64)


def compute_nested_anova(
    records: LocalClusterRecords,
    labels: torch.Tensor,
) -> NestedAnovaResult:
    """Closed-form nested-ANOVA on a record table with a global assignment.

    The decomposition uses size-weighted means at every level so the
    identity ``SS_total = within + between_in + between`` holds to FP64
    precision regardless of how the global labels were obtained.
    """

    if records.num_records == 0:
        raise ValueError("Cannot compute nested ANOVA on empty records.")
    if labels.shape[0] != records.num_records:
        raise ValueError(
            f"labels length {labels.shape[0]} != num_records {records.num_records}"
        )

    labels = labels.to(dtype=torch.int64)
    centroids = records.centroids                                  # (M, D) f64
    sizes = records.sizes.to(dtype=torch.float64)                  # (M,)
    within_ss = records.within_ss                                  # (M,) f64
    D = records.embedding_dim

    total_tokens = float(sizes.sum().item())
    if total_tokens == 0:
        raise ValueError("Total token count is zero.")

    # Global mean (size-weighted across all local centroids).
    mu_G = (centroids * sizes.unsqueeze(1)).sum(dim=0) / total_tokens          # (D,)

    # Per-global-cluster size-weighted mean.
    unique_labels, inverse = labels.unique(return_inverse=True)
    G = int(unique_labels.shape[0])
    weighted_sum_per_g = torch.zeros(G, D, dtype=torch.float64)
    weight_per_g = torch.zeros(G, dtype=torch.float64)
    weighted_sum_per_g.index_add_(0, inverse, centroids * sizes.unsqueeze(1))
    weight_per_g.index_add_(0, inverse, sizes)
    if (weight_per_g <= 0).any():
        raise RuntimeError("Empty global cluster encountered.")
    mu_g = weighted_sum_per_g / weight_per_g.unsqueeze(1)                       # (G, D)

    # SS components.
    ss_within_local = float(within_ss.sum().item())

    diffs_local = centroids - mu_g[inverse]                                     # (M, D)
    ss_between_local_in_global = float(((diffs_local * diffs_local).sum(dim=1) * sizes).sum().item())

    diffs_global = mu_g - mu_G.unsqueeze(0)                                     # (G, D)
    ss_between_global = float(((diffs_global * diffs_global).sum(dim=1) * weight_per_g).sum().item())

    ss_total = ss_within_local + ss_between_local_in_global + ss_between_global

    return NestedAnovaResult(
        within_local=ss_within_local / total_tokens,
        between_local_in_global=ss_between_local_in_global / total_tokens,
        between_global=ss_between_global / total_tokens,
        total=ss_total / total_tokens,
        num_tokens=int(total_tokens),
        num_local_clusters=records.num_records,
        num_global_clusters=G,
    )


def compute_nested_anova_total_from_tokens(
    tokens_BTD: torch.Tensor,
    *,
    image_ids: Sequence[int] | torch.Tensor | None = None,
) -> float:
    """Reference SS_total / N_tokens computed directly from raw normalized tokens.

    Useful as an end-to-end consistency check against
    :func:`compute_nested_anova`. The two values should agree to FP64
    noise: the record-based decomposition uses size-weighted means of
    local centroids, which (for fixed-length sequences with no token
    discarded) coincides with the token-level grand mean.
    """

    if tokens_BTD.ndim != 3:
        raise ValueError(f"Expected (B, T, D), got {tuple(tokens_BTD.shape)}")
    x = _to_cpu_f64(tokens_BTD)                                                  # (B, T, D)
    flat = x.reshape(-1, x.shape[-1])
    mu = flat.mean(dim=0)
    return float((flat - mu).pow(2).sum(dim=1).mean().item())
