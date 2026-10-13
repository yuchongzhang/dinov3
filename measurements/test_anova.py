from __future__ import annotations

import unittest

import torch

from measurements.anova import (
    TokenMeanAccumulator,
    compute_weighted_token_anova,
    exclude_cls_token,
)


def manual_weighted_anova(tokens: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    tokens = tokens.to(dtype=torch.float64)
    targets = targets.reshape(-1)
    num_sequences, tokens_per_sequence, _ = tokens.shape

    sequence_means = tokens.mean(dim=1)
    class_means = {}
    class_counts = {}
    for class_id in targets.unique().tolist():
        class_id = int(class_id)
        mask = targets == class_id
        class_means[class_id] = sequence_means[mask].mean(dim=0)
        class_counts[class_id] = int(mask.sum().item())

    global_mean = sequence_means.mean(dim=0)
    within_seq = (tokens - sequence_means.unsqueeze(1)).pow(2).sum() / (num_sequences * tokens_per_sequence)

    within_class_sum = 0.0
    between_class_sum = 0.0
    for class_id, class_mean in class_means.items():
        mask = targets == class_id
        within_class_sum += (sequence_means[mask] - class_mean).pow(2).sum()
        between_class_sum += class_counts[class_id] * (class_mean - global_mean).pow(2).sum()

    within_class = within_class_sum / num_sequences
    between_class = between_class_sum / num_sequences
    total = (tokens - global_mean).pow(2).sum() / (num_sequences * tokens_per_sequence)

    return {
        "within_seq": float(within_seq.item()),
        "within_class": float(within_class.item()),
        "between_class": float(between_class.item()),
        "total": float(total.item()),
    }


class WeightedTokenAnovaTest(unittest.TestCase):
    def assertAnovaMatchesManual(self, tokens: torch.Tensor, targets: torch.Tensor) -> None:
        expected = manual_weighted_anova(tokens, targets)
        actual = compute_weighted_token_anova(tokens, targets)

        self.assertAlmostEqual(actual.within_seq, expected["within_seq"], places=12)
        self.assertAlmostEqual(actual.within_class, expected["within_class"], places=12)
        self.assertAlmostEqual(actual.between_class, expected["between_class"], places=12)
        self.assertAlmostEqual(actual.total, expected["total"], places=12)
        self.assertAlmostEqual(actual.total, actual.components_sum, places=12)

    def test_balanced_classes(self) -> None:
        tokens = torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
                [[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]],
                [[10.0, 0.0], [12.0, 0.0], [14.0, 0.0]],
                [[11.0, 1.0], [13.0, 1.0], [15.0, 1.0]],
            ]
        )
        targets = torch.tensor([0, 0, 1, 1])
        self.assertAnovaMatchesManual(tokens, targets)

    def test_imbalanced_classes(self) -> None:
        tokens = torch.tensor(
            [
                [[0.0, 0.0], [2.0, 1.0], [4.0, 2.0]],
                [[10.0, 0.0], [12.0, 1.0], [14.0, 2.0]],
                [[11.0, 1.0], [13.0, 2.0], [15.0, 3.0]],
            ]
        )
        targets = torch.tensor([0, 1, 1])
        self.assertAnovaMatchesManual(tokens, targets)

    def test_compute_can_exclude_cls_token(self) -> None:
        tokens = torch.tensor(
            [
                [[100.0, 100.0], [0.0, 0.0], [2.0, 0.0]],
                [[200.0, 200.0], [1.0, 1.0], [3.0, 1.0]],
                [[300.0, 300.0], [10.0, 0.0], [12.0, 0.0]],
                [[400.0, 400.0], [11.0, 1.0], [13.0, 1.0]],
            ]
        )
        targets = torch.tensor([0, 0, 1, 1])

        expected = manual_weighted_anova(tokens[:, 1:, :], targets)
        actual = compute_weighted_token_anova(tokens, targets, exclude_cls=True)

        self.assertAlmostEqual(actual.within_seq, expected["within_seq"], places=12)
        self.assertAlmostEqual(actual.within_class, expected["within_class"], places=12)
        self.assertAlmostEqual(actual.between_class, expected["between_class"], places=12)
        self.assertAlmostEqual(actual.total, expected["total"], places=12)

    def test_exclude_cls_requires_patch_tokens(self) -> None:
        with self.assertRaises(ValueError):
            exclude_cls_token(torch.zeros(2, 1, 3))

    def test_rejects_variable_length_batches(self) -> None:
        accumulator = TokenMeanAccumulator()
        accumulator.update(torch.zeros(1, 2, 3), torch.tensor([0]))
        with self.assertRaises(ValueError):
            accumulator.update(torch.zeros(1, 3, 3), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
