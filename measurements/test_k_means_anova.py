from __future__ import annotations

import unittest

import torch

from measurements.k_means_anova import (
    TokenPCAAccumulator,
    cluster_image_tokens_kmeans,
    compute_k_means_anova,
    fit_global_kmeans,
    project_tokens_onto_pca,
)


class KMeansAnovaTest(unittest.TestCase):
    def test_kmeans_anova_identity_holds(self) -> None:
        tokens = torch.tensor(
            [
                [[-2.0, 0.0], [-1.8, 0.0], [1.8, 0.0], [2.0, 0.0]],
                [[-2.0, 4.0], [-1.8, 4.0], [1.8, 4.0], [2.0, 4.0]],
            ]
        )

        records = cluster_image_tokens_kmeans(tokens, k_local=2, seed=0)
        _centers, labels = fit_global_kmeans(records, k_global=2, seed=0)
        result = compute_k_means_anova(records, labels)

        self.assertEqual(records.num_records, 4)
        self.assertEqual(result.num_global_clusters, 2)
        self.assertLess(result.absolute_anova_error, 1e-12)
        self.assertAlmostEqual(result.total, result.components_sum, places=12)

    def test_pca_projection_before_kmeans_changes_record_dimension(self) -> None:
        tokens = torch.tensor(
            [
                [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            ]
        )
        accumulator = TokenPCAAccumulator()
        accumulator.update(tokens)
        projection = accumulator.compute(top_k=1)

        projected = project_tokens_onto_pca(tokens, projection)
        records = cluster_image_tokens_kmeans(projected, k_local=2, seed=0)

        self.assertEqual(tuple(projected.shape), (1, 4, 1))
        self.assertEqual(records.embedding_dim, 1)


if __name__ == "__main__":
    unittest.main()
