"""Segmentation-driven ANOVA on DINOv3 token embeddings.

Same accumulator/dataclass conventions as ``measurements.nested_anova``,
but the *local* clusters are not learned — they come straight from a
segmentation map (e.g. ADE20K). Each (image, segmentation-class) pair
is one local cluster.

The driver can optionally fit a PCA basis over the non-ignored tokens and
project each embedding to the top-k components before these sufficient
statistics are formed.

Two decompositions are supported:

* **Two-level** (no global clustering):
    ``SS_total = SS_within_local + SS_between_local``
* **Three-level** (with global clustering): same identity as
  ``compute_nested_anova`` — pass the records produced here through
  the existing :func:`measurements.nested_anova.compute_nested_anova`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from measurements.nested_anova import LocalClusterRecords


@dataclass(frozen=True)
class TokenPCAProjection:
    """Top-k PCA projection fitted from token embeddings.

    ``components`` has shape ``(K, D)`` and contains principal axes as rows.
    The covariance convention is population-scaled (divide by ``N``), matching
    the ANOVA code's SS / token-count variance convention.
    """

    mean: torch.Tensor
    components: torch.Tensor
    explained_variance: torch.Tensor
    total_variance: float
    num_tokens: int

    @property
    def num_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[1])

    @property
    def retained_variance(self) -> float:
        return float(self.explained_variance.sum().item())

    @property
    def retained_variance_ratio(self) -> float:
        if self.total_variance <= 0:
            return 0.0
        return self.retained_variance / self.total_variance


@dataclass
class TokenPCAAccumulator:
    """Streaming sufficient statistics for PCA over token embeddings."""

    count: int = 0
    feature_sum: torch.Tensor | None = None
    gram_sum: torch.Tensor | None = None

    def update(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Accumulate tokens of shape ``(B, T, D)``.

        If ``mask`` is supplied, only entries where ``mask`` is true are used.
        This lets the segmentation driver fit PCA on exactly the non-ignored
        tokens that enter the variance decomposition.
        """

        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens (B, T, D), got {tuple(tokens.shape)}")
        B, T, D = tokens.shape
        if mask is not None:
            if mask.shape != (B, T):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != tokens (B, T) ({B}, {T})"
                )
            flat = tokens[mask.to(device=tokens.device, dtype=torch.bool)]
        else:
            flat = tokens.reshape(B * T, D)
        if flat.numel() == 0:
            return

        flat64 = flat.detach().to(device="cpu", dtype=torch.float64)
        if flat64.shape[1] != D:
            raise RuntimeError("Flattened token feature dimension changed unexpectedly.")
        if self.feature_sum is not None and self.feature_sum.shape[0] != D:
            raise ValueError(
                f"Cannot accumulate tokens with D={D}; existing PCA stats have D={self.feature_sum.shape[0]}."
            )

        batch_sum = flat64.sum(dim=0)
        batch_gram = flat64.T @ flat64
        if self.feature_sum is None:
            self.feature_sum = batch_sum
            self.gram_sum = batch_gram
        else:
            self.feature_sum += batch_sum
            self.gram_sum += batch_gram
        self.count += int(flat64.shape[0])

    def compute(self, top_k: int) -> TokenPCAProjection:
        """Return the top-k PCA axes from the accumulated token statistics."""

        if self.count == 0 or self.feature_sum is None or self.gram_sum is None:
            raise ValueError("Cannot compute PCA: no tokens were accumulated.")
        D = int(self.feature_sum.shape[0])
        if top_k < 1 or top_k > D:
            raise ValueError(f"top_k must be in [1, D={D}], got {top_k}.")

        mean = self.feature_sum / float(self.count)
        covariance = self.gram_sum / float(self.count) - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        eigvals, eigvecs = torch.linalg.eigh(covariance)
        order = torch.argsort(eigvals, descending=True)
        top = order[:top_k]
        components = eigvecs[:, top].T.contiguous()

        explained = eigvals[top].clamp_min(0.0).contiguous()
        total_variance = float(eigvals.clamp_min(0.0).sum().item())
        return TokenPCAProjection(
            mean=mean.contiguous(),
            components=components.to(dtype=torch.float64),
            explained_variance=explained.to(dtype=torch.float64),
            total_variance=total_variance,
            num_tokens=self.count,
        )


def project_tokens_onto_pca(
    tokens: torch.Tensor,
    projection: TokenPCAProjection,
) -> torch.Tensor:
    """Center and project ``(B, T, D)`` tokens onto a fitted PCA basis."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens (B, T, D), got {tuple(tokens.shape)}")
    D = tokens.shape[-1]
    if projection.mean.shape != (D,) or projection.components.shape[1] != D:
        raise ValueError(
            "PCA projection is incompatible with token dimension "
            f"D={D}: mean={tuple(projection.mean.shape)}, components={tuple(projection.components.shape)}."
        )

    work_dtype = torch.float64 if tokens.dtype == torch.float64 else torch.float32
    mean = projection.mean.to(device=tokens.device, dtype=work_dtype)
    components = projection.components.to(device=tokens.device, dtype=work_dtype)
    centered = tokens.to(dtype=work_dtype) - mean
    return centered @ components.T


def pool_segmentation_to_patches(
    seg: torch.Tensor,
    *,
    patch_size: int,
    ignore_label: int,
) -> torch.Tensor:
    """Majority-pool a per-pixel segmentation map down to a per-patch grid.

    For each ``patch_size × patch_size`` block, the patch label is the
    most common *non-ignored* class. If every pixel in a patch is
    ``ignore_label``, the patch label is ``ignore_label``.

    Parameters
    ----------
    seg : (B, H, W) integer tensor
        Per-pixel class indices. ``H`` and ``W`` must be divisible by
        ``patch_size``.
    patch_size : int
        Edge length of the square pooling window.
    ignore_label : int
        Pixels equal to this label are excluded from the per-patch mode.
        Pass a negative integer (e.g. ``-1``) to keep all pixels.

    Returns
    -------
    pooled : (B, T) int64 tensor with ``T = (H/patch_size)*(W/patch_size)``.
    """

    if seg.ndim != 3:
        raise ValueError(f"Expected (B, H, W) segmentation, got shape {tuple(seg.shape)}")
    B, H, W = seg.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"H={H} W={W} not divisible by patch_size={patch_size}"
        )
    Hp, Wp = H // patch_size, W // patch_size
    seg64 = seg.detach().to(device="cpu", dtype=torch.int64)

    # (B, Hp, p, Wp, p) -> (B, Hp, Wp, p*p) -> (N_patches, p*p)
    blocks = (
        seg64.reshape(B, Hp, patch_size, Wp, patch_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(B * Hp * Wp, patch_size * patch_size)
    )

    # Build a non-negative class axis. Determine the max valid label across
    # the batch (negative labels are out-of-range and ignored).
    valid_labels = blocks[blocks >= 0]
    max_label = int(valid_labels.max().item()) if valid_labels.numel() > 0 else 0
    sentinel = max_label + 1                                # for ignored / out-of-range

    safe = blocks.clone()
    if ignore_label >= 0:
        safe[safe == ignore_label] = sentinel
    safe[safe < 0] = sentinel
    safe[safe > sentinel] = sentinel                        # defensive

    counts = torch.zeros(blocks.shape[0], sentinel + 1, dtype=torch.int64)
    ones = torch.ones_like(safe)
    counts.scatter_add_(1, safe, ones)
    counts[:, sentinel] = 0                                  # never pick the sentinel
    if 0 <= ignore_label <= max_label:
        counts[:, ignore_label] = 0                          # belt-and-suspenders

    has_valid = counts.sum(dim=1) > 0
    pooled = counts.argmax(dim=1)
    pooled[~has_valid] = ignore_label
    return pooled.reshape(B, Hp * Wp).to(torch.int64)


def cluster_image_tokens_from_segmentation(
    tokens: torch.Tensor,
    patch_labels: torch.Tensor,
    *,
    image_ids: Sequence[int] | torch.Tensor | None = None,
    ignore_label: int,
) -> LocalClusterRecords:
    """Build :class:`LocalClusterRecords` directly from segmentation labels.

    Parameters
    ----------
    tokens : (B, T, D) float tensor
        L2-normalized DINOv3 last-layer patch tokens.
    patch_labels : (B, T) int64 tensor
        Per-patch class index, e.g. the output of
        :func:`pool_segmentation_to_patches`.
    image_ids : optional length-B sequence
        Stable identifiers (defaults to ``range(B)``).
    ignore_label : int
        Patches with this label are dropped (no cluster row emitted).
    """

    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens (B, T, D), got {tuple(tokens.shape)}")
    B, T, D = tokens.shape
    if patch_labels.shape != (B, T):
        raise ValueError(
            f"patch_labels shape {tuple(patch_labels.shape)} != tokens (B, T) ({B}, {T})"
        )

    tokens_cpu = tokens.detach().to(device="cpu", dtype=torch.float64)
    labels_cpu = patch_labels.detach().to(device="cpu", dtype=torch.int64)

    if image_ids is None:
        image_ids_t = torch.arange(B, dtype=torch.int64)
    else:
        image_ids_t = torch.as_tensor(image_ids, dtype=torch.int64)
        if image_ids_t.shape[0] != B:
            raise ValueError(f"image_ids length {image_ids_t.shape[0]} != B={B}")

    out_image_ids: list[int] = []
    out_sizes: list[int] = []
    out_centroids: list[torch.Tensor] = []
    out_within: list[float] = []

    for b in range(B):
        x = tokens_cpu[b]                                            # (T, D)
        l = labels_cpu[b]                                            # (T,)
        unique = torch.unique(l).tolist()
        for lbl in unique:
            if lbl == ignore_label:
                continue
            mask = (l == lbl)
            n_l = int(mask.sum().item())
            members = x[mask]
            centroid = members.mean(dim=0)
            within = (members - centroid.unsqueeze(0)).pow(2).sum().item()
            out_image_ids.append(int(image_ids_t[b].item()))
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


@dataclass(frozen=True)
class TwoLevelAnovaResult:
    """Empirical 2-level ANOVA variances (each = SS_component / total_tokens)."""

    within_local: float
    between_local: float
    total: float
    num_tokens: int
    num_local_clusters: int

    @property
    def components_sum(self) -> float:
        return self.within_local + self.between_local

    @property
    def anova_error(self) -> float:
        return self.total - self.components_sum

    @property
    def absolute_anova_error(self) -> float:
        return abs(self.anova_error)

    def variances(self) -> dict[str, float]:
        return {
            "within_local": self.within_local,
            "between_local": self.between_local,
            "total": self.total,
        }

    def percentages(self) -> dict[str, float]:
        if self.total == 0:
            return {k: 0.0 for k in ("within_local", "between_local", "components_sum")}
        return {
            "within_local": 100.0 * self.within_local / self.total,
            "between_local": 100.0 * self.between_local / self.total,
            "components_sum": 100.0 * self.components_sum / self.total,
        }


def compute_two_level_anova(records: LocalClusterRecords) -> TwoLevelAnovaResult:
    """Closed-form 2-level ANOVA: ``SS_total = within_local + between_local``.

    Each cluster row in ``records`` is treated as one group. Means are
    size-weighted so the cross-term vanishes and the identity holds to
    FP64 precision.
    """

    if records.num_records == 0:
        raise ValueError("Cannot compute two-level ANOVA on empty records.")

    centroids = records.centroids                                   # (M, D) f64
    sizes = records.sizes.to(dtype=torch.float64)                   # (M,)
    within_ss = records.within_ss                                   # (M,) f64

    total_tokens = float(sizes.sum().item())
    if total_tokens == 0:
        raise ValueError("Total token count is zero.")

    mu_G = (centroids * sizes.unsqueeze(1)).sum(dim=0) / total_tokens

    diffs = centroids - mu_G.unsqueeze(0)
    ss_between = float(((diffs * diffs).sum(dim=1) * sizes).sum().item())
    ss_within = float(within_ss.sum().item())
    ss_total = ss_within + ss_between

    return TwoLevelAnovaResult(
        within_local=ss_within / total_tokens,
        between_local=ss_between / total_tokens,
        total=ss_total / total_tokens,
        num_tokens=int(total_tokens),
        num_local_clusters=records.num_records,
    )
