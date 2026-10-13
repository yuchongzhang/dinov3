from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TokenAnovaMeans:
    """Weighted token ANOVA means for fixed-length token sequences."""

    global_mean: torch.Tensor
    class_means: dict[int, torch.Tensor]
    seq_counts_per_class: dict[int, int]
    token_counts_per_class: dict[int, int]
    tokens_per_sequence: int

    @property
    def num_sequences(self) -> int:
        return sum(self.seq_counts_per_class.values())

    @property
    def num_tokens(self) -> int:
        return sum(self.token_counts_per_class.values())

    @property
    def num_classes(self) -> int:
        return len(self.seq_counts_per_class)


@dataclass(frozen=True)
class TokenAnovaResult:
    """Weighted empirical token ANOVA variances."""

    within_seq: float
    within_class: float
    between_class: float
    total: float

    @property
    def components_sum(self) -> float:
        return self.within_seq + self.within_class + self.between_class

    @property
    def anova_error(self) -> float:
        return self.total - self.components_sum

    @property
    def absolute_anova_error(self) -> float:
        return abs(self.anova_error)

    def percentages(self) -> dict[str, float]:
        if self.total == 0:
            return {
                "within_seq": 0.0,
                "within_class": 0.0,
                "between_class": 0.0,
                "components_sum": 0.0,
            }
        return {
            "within_seq": 100.0 * self.within_seq / self.total,
            "within_class": 100.0 * self.within_class / self.total,
            "between_class": 100.0 * self.between_class / self.total,
            "components_sum": 100.0 * self.components_sum / self.total,
        }

    def variances(self) -> dict[str, float]:
        return {
            "within_seq": self.within_seq,
            "within_class": self.within_class,
            "between_class": self.between_class,
            "total": self.total,
        }


def _check_tokens_and_targets(tokens: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, D), got {tuple(tokens.shape)}")
    if tokens.shape[1] == 0:
        raise ValueError("Expected at least one token per sequence.")
    if targets.ndim != 1:
        targets = targets.reshape(-1)
    if targets.shape[0] != tokens.shape[0]:
        raise ValueError(f"Expected one target per sequence, got {targets.shape[0]} targets for {tokens.shape[0]} sequences")
    return tokens, targets.to(device=tokens.device)


def exclude_cls_token(tokens: torch.Tensor) -> torch.Tensor:
    """Return patch-token features when the leading token is the CLS token."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, T, D), got {tuple(tokens.shape)}")
    if tokens.shape[1] <= 1:
        raise ValueError("Cannot exclude CLS token from a sequence with no remaining patch tokens.")
    return tokens[:, 1:, :]


def _as_float64(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.detach().to(dtype=torch.float64)


class TokenMeanAccumulator:
    """
    First pass for weighted empirical token ANOVA.

    The definitions use sequence means mu_{i,c}. For fixed-length sequences this
    is equivalent to accumulating token means directly, but the sequence-mean
    form makes the intended WithinClassVar and BetweenClassVar weighting explicit.
    """

    def __init__(self) -> None:
        self.sequence_mean_sums_per_class: dict[int, torch.Tensor] = {}
        self.seq_counts_per_class: dict[int, int] = defaultdict(int)
        self.token_counts_per_class: dict[int, int] = defaultdict(int)
        self.tokens_per_sequence: int | None = None

    def update(self, tokens: torch.Tensor, targets: torch.Tensor) -> None:
        tokens, targets = _check_tokens_and_targets(tokens, targets)
        if self.tokens_per_sequence is None:
            self.tokens_per_sequence = tokens.shape[1]
        elif tokens.shape[1] != self.tokens_per_sequence:
            raise ValueError(
                "TokenAnova currently expects fixed-length sequences. "
                f"Expected T={self.tokens_per_sequence}, got T={tokens.shape[1]}."
            )

        tokens64 = _as_float64(tokens)
        sequence_means = tokens64.mean(dim=1)
        for class_id in targets.detach().cpu().unique().tolist():
            class_id = int(class_id)
            mask = targets == class_id
            class_sequence_means = sequence_means[mask]
            class_sum = class_sequence_means.sum(dim=0).cpu()
            self.sequence_mean_sums_per_class[class_id] = (
                self.sequence_mean_sums_per_class.get(class_id, torch.zeros_like(class_sum)) + class_sum
            )
            class_seq_count = int(mask.sum().item())
            self.seq_counts_per_class[class_id] += class_seq_count
            self.token_counts_per_class[class_id] += class_seq_count * tokens.shape[1]

    def finalize(self) -> TokenAnovaMeans:
        if not self.sequence_mean_sums_per_class or self.tokens_per_sequence is None:
            raise RuntimeError("Cannot finalize ANOVA means before seeing any batches.")

        total_sequences = sum(self.seq_counts_per_class.values())
        if total_sequences == 0:
            raise RuntimeError("Cannot finalize ANOVA means with zero sequences.")

        class_means = {
            class_id: class_sum / self.seq_counts_per_class[class_id]
            for class_id, class_sum in sorted(self.sequence_mean_sums_per_class.items())
        }
        global_mean = sum(self.sequence_mean_sums_per_class.values()) / total_sequences

        return TokenAnovaMeans(
            global_mean=global_mean.cpu(),
            class_means={class_id: class_mean.cpu() for class_id, class_mean in class_means.items()},
            seq_counts_per_class=dict(sorted(self.seq_counts_per_class.items())),
            token_counts_per_class=dict(sorted(self.token_counts_per_class.items())),
            tokens_per_sequence=self.tokens_per_sequence,
        )


def compute_between_class_var(
    global_mean: torch.Tensor,
    class_means: Mapping[int, torch.Tensor],
    counts_per_class: Mapping[int, int],
) -> float:
    """Compute sum_c count_c ||mu_c - mu_G||^2 / sum_c count_c."""

    total_count = sum(counts_per_class.values())
    if total_count == 0:
        raise ValueError("counts_per_class must contain at least one sample.")

    global_mean = global_mean.detach().to(dtype=torch.float64, device="cpu")
    between_class_sum = 0.0
    for class_id, count in counts_per_class.items():
        class_mean = class_means[class_id].detach().to(dtype=torch.float64, device="cpu")
        between_class_sum += float(((class_mean - global_mean).pow(2).sum()).item()) * (count / total_count)
    return between_class_sum


class TokenVarianceAccumulator:
    """Second pass for weighted empirical token ANOVA."""

    def __init__(self, means: TokenAnovaMeans) -> None:
        self.means = means
        self.within_seq_sum = 0.0
        self.within_class_sum = 0.0
        self.total_sum = 0.0
        self.seq_counts_per_class: dict[int, int] = defaultdict(int)
        self.token_counts_per_class: dict[int, int] = defaultdict(int)

    def update(self, tokens: torch.Tensor, targets: torch.Tensor) -> None:
        tokens, targets = _check_tokens_and_targets(tokens, targets)
        if tokens.shape[1] != self.means.tokens_per_sequence:
            raise ValueError(
                "TokenAnova currently expects fixed-length sequences. "
                f"Expected T={self.means.tokens_per_sequence}, got T={tokens.shape[1]}."
            )

        tokens64 = _as_float64(tokens)
        sequence_means = tokens64.mean(dim=1)
        global_mean = self.means.global_mean.to(device=tokens.device, dtype=torch.float64)

        for class_id in targets.detach().cpu().unique().tolist():
            class_id = int(class_id)
            if class_id not in self.means.class_means:
                raise ValueError(f"Class {class_id} was seen in variance pass but not in mean pass.")

            mask = targets == class_id
            class_tokens = tokens64[mask]
            class_sequence_means = sequence_means[mask]
            class_mean = self.means.class_means[class_id].to(device=tokens.device, dtype=torch.float64)

            self.within_seq_sum += float(
                (class_tokens - class_sequence_means.unsqueeze(1)).pow(2).sum(dim=-1).sum().item()
            )
            self.within_class_sum += float((class_sequence_means - class_mean).pow(2).sum(dim=-1).sum().item())
            self.total_sum += float((class_tokens - global_mean).pow(2).sum(dim=-1).sum().item())

            class_seq_count = int(mask.sum().item())
            self.seq_counts_per_class[class_id] += class_seq_count
            self.token_counts_per_class[class_id] += class_seq_count * tokens.shape[1]

    def finalize(self) -> TokenAnovaResult:
        if dict(sorted(self.seq_counts_per_class.items())) != self.means.seq_counts_per_class:
            raise RuntimeError("Mean and variance passes saw different per-class sequence counts.")
        if dict(sorted(self.token_counts_per_class.items())) != self.means.token_counts_per_class:
            raise RuntimeError("Mean and variance passes saw different per-class token counts.")

        total_sequences = self.means.num_sequences
        total_tokens = self.means.num_tokens
        if total_sequences == 0 or total_tokens == 0:
            raise RuntimeError("Cannot finalize ANOVA variances with zero samples.")

        return TokenAnovaResult(
            within_seq=self.within_seq_sum / total_tokens,
            within_class=self.within_class_sum / total_sequences,
            between_class=compute_between_class_var(
                self.means.global_mean,
                self.means.class_means,
                self.means.seq_counts_per_class,
            ),
            total=self.total_sum / total_tokens,
        )


def compute_weighted_token_anova(
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    exclude_cls: bool = False,
) -> TokenAnovaResult:
    """Convenience helper for in-memory tensors with shape (N, T, D)."""

    if exclude_cls:
        tokens = exclude_cls_token(tokens)

    mean_accumulator = TokenMeanAccumulator()
    mean_accumulator.update(tokens, targets)
    means = mean_accumulator.finalize()

    variance_accumulator = TokenVarianceAccumulator(means)
    variance_accumulator.update(tokens, targets)
    return variance_accumulator.finalize()


class TokenMeanHook:
    """
    Backward-compatible hook wrapper for the first ANOVA pass.

    New code should prefer TokenMeanAccumulator directly.
    """

    def __init__(self) -> None:
        self.accumulator = TokenMeanAccumulator()
        self.handle = None
        self.means: TokenAnovaMeans | None = None

    def get_mean_hook_function(self, targets: torch.Tensor) -> Callable:
        def hook_function(module, input, output):
            del module, input
            self.accumulator.update(output, targets)

        return hook_function

    def compute_means(self):
        self.means = self.accumulator.finalize()
        return self.means.global_mean, self.means.class_means

    @property
    def token_counts_per_class(self) -> dict[int, int]:
        return dict(self.accumulator.token_counts_per_class)

    def register_hook(self, model: nn.Module, targets: torch.Tensor) -> None:
        self.handle = model.norm.register_forward_hook(self.get_mean_hook_function(targets))

    def remove_handle(self) -> None:
        if self.handle is not None:
            self.handle.remove()


class TokenVarHook:
    """
    Backward-compatible hook wrapper for the variance pass.

    New code should prefer TokenVarianceAccumulator directly.
    """

    def __init__(self, class_means, global_mean, token_counts_per_class, device) -> None:
        self.class_means = {int(k): v.detach().to(device=device, dtype=torch.float64) for k, v in class_means.items()}
        self.global_mean = global_mean.detach().to(device=device, dtype=torch.float64)
        self.token_counts_per_class = dict(token_counts_per_class)
        self.device = device
        self.handle = None

        self.within_seq_sum_per_class = defaultdict(float)
        self.within_class_sum_per_class = defaultdict(float)
        self.seq_counts_per_class = defaultdict(int)
        self.total_sum = 0.0

    def get_var_hook_function(self, targets: torch.Tensor) -> Callable:
        def hook_function(module, input, output):
            del module, input
            tokens, target_labels = _check_tokens_and_targets(output, targets)
            tokens64 = _as_float64(tokens)
            sequence_means = tokens64.mean(dim=1)
            for class_id in target_labels.detach().cpu().unique().tolist():
                class_id = int(class_id)
                mask = target_labels == class_id
                class_tokens = tokens64[mask]
                class_sequence_means = sequence_means[mask]
                class_mean = self.class_means[class_id].to(device=tokens.device, dtype=torch.float64)

                self.within_seq_sum_per_class[class_id] += float(
                    (class_tokens - class_sequence_means.unsqueeze(1)).pow(2).sum(dim=-1).sum().item()
                )
                self.within_class_sum_per_class[class_id] += float(
                    (class_sequence_means - class_mean).pow(2).sum(dim=-1).sum().item()
                )
                self.total_sum += float((class_tokens - self.global_mean).pow(2).sum(dim=-1).sum().item())
                self.seq_counts_per_class[class_id] += int(mask.sum().item())

        return hook_function

    def compute_var(self):
        total_tokens = sum(self.token_counts_per_class.values())
        total_sequences = sum(self.seq_counts_per_class.values())
        if total_tokens == 0 or total_sequences == 0:
            raise RuntimeError("Cannot compute ANOVA variances before seeing any batches.")

        within_seq_var = sum(self.within_seq_sum_per_class.values()) / total_tokens
        within_class_var = sum(self.within_class_sum_per_class.values()) / total_sequences
        total_var = self.total_sum / total_tokens
        return total_var, within_class_var, within_seq_var

    def register_hook(self, model: nn.Module, targets: torch.Tensor) -> None:
        self.handle = model.norm.register_forward_hook(self.get_var_hook_function(targets))

    def remove_handle(self) -> None:
        if self.handle is not None:
            self.handle.remove()
