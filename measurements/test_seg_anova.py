from __future__ import annotations

import unittest

import torch

from measurements.seg_anova import (
    TokenPCAAccumulator,
    cluster_image_tokens_from_segmentation,
    compute_two_level_anova,
    project_tokens_onto_pca,
)


class SegmentationPcaAnovaTest(unittest.TestCase):
    def test_pca_accumulator_matches_top_centered_eigenvalue(self) -> None:
        tokens = torch.tensor(
            [
                [[-2.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [2.0, 0.0], [0.0, 50.0]],
            ]
        )
        mask = torch.tensor(
            [
                [True, True, True],
                [True, True, False],
            ]
        )

        accumulator = TokenPCAAccumulator()
        accumulator.update(tokens, mask=mask)
        projection = accumulator.compute(top_k=1)
        projected = project_tokens_onto_pca(tokens, projection)

        selected = tokens[mask].double()
        centered = selected - selected.mean(dim=0)
        covariance = centered.T @ centered / selected.shape[0]
        expected_top_variance = torch.linalg.eigvalsh(covariance).max().item()
        actual_top_variance = projected[mask].double().square().mean().item()

        self.assertEqual(tuple(projected.shape), (2, 3, 1))
        self.assertAlmostEqual(actual_top_variance, expected_top_variance, places=12)
        self.assertAlmostEqual(projection.explained_variance[0].item(), expected_top_variance, places=12)

    def test_pca_projected_records_preserve_two_level_total(self) -> None:
        tokens = torch.tensor(
            [
                [
                    [-2.0, 0.0],
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                ]
            ]
        )
        labels = torch.tensor([[1, 1, 2, 2]])

        accumulator = TokenPCAAccumulator()
        accumulator.update(tokens)
        projection = accumulator.compute(top_k=1)
        projected = project_tokens_onto_pca(tokens, projection)

        records = cluster_image_tokens_from_segmentation(
            projected,
            labels,
            ignore_label=0,
        )
        result = compute_two_level_anova(records)

        flat = projected.reshape(-1, 1).double()
        expected_total = (flat - flat.mean(dim=0)).square().sum().item() / flat.shape[0]

        self.assertEqual(records.embedding_dim, 1)
        self.assertLess(result.absolute_anova_error, 1e-12)
        self.assertAlmostEqual(result.total, expected_total, places=12)


if __name__ == "__main__":
    unittest.main()
