from __future__ import annotations

import unittest

import torch

from measurements.anova import compute_weighted_token_anova
from measurements.nested_anova import (
    LocalClusterRecords,
    assign_global,
    cluster_image_tokens,
    compute_nested_anova,
    fit_global_centers,
    l2_normalize_tokens,
)


def make_planted_tokens(
    *,
    n_global: int = 3,
    n_local_per_global: int = 2,
    n_images_per_local: int = 4,
    tokens_per_local: int = 8,
    D: int = 12,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers_g = torch.randn(n_global, D, generator=g, dtype=torch.float64)
    centers_g = centers_g / centers_g.norm(dim=-1, keepdim=True)
    images: list[torch.Tensor] = []
    image_globals: list[int] = []
    for gi in range(n_global):
        for _li in range(n_local_per_global):
            offset = 0.4 * torch.randn(D, generator=g, dtype=torch.float64)
            local_a = (centers_g[gi] + offset)
            local_b = (centers_g[gi] - offset)
            local_a = local_a / local_a.norm()
            local_b = local_b / local_b.norm()
            for _ in range(n_images_per_local):
                half = tokens_per_local // 2
                a = local_a + noise * torch.randn(half, D, generator=g, dtype=torch.float64)
                b = local_b + noise * torch.randn(tokens_per_local - half, D, generator=g, dtype=torch.float64)
                images.append(torch.cat([a, b], dim=0))
                image_globals.append(gi)
    tokens = l2_normalize_tokens(torch.stack(images, dim=0))
    return tokens, torch.tensor(image_globals, dtype=torch.int64)


class NestedAnovaTest(unittest.TestCase):
    def test_ss_identity_holds(self) -> None:
        tokens, _ = make_planted_tokens()
        records = cluster_image_tokens(tokens, distance_threshold=0.5)
        centers = fit_global_centers(records, k_global=3, sample_size=None, seed=0)
        labels = assign_global(records, centers)
        result = compute_nested_anova(records, labels)
        self.assertLess(result.absolute_anova_error, 1e-10)
        self.assertAlmostEqual(result.components_sum, result.total, places=10)

    def test_planted_structure_recovered(self) -> None:
        tokens, _ = make_planted_tokens()
        records = cluster_image_tokens(tokens, distance_threshold=0.5)
        # Each image has 8 tokens, planted as 2 local clusters.
        counts: dict[int, int] = {}
        for img in records.image_ids.tolist():
            counts[img] = counts.get(img, 0) + 1
        self.assertTrue(all(v == 2 for v in counts.values()),
                        f"expected exactly 2 clusters per image, got {sorted(set(counts.values()))}")

    def test_reduces_to_existing_anova(self) -> None:
        """K_local=1 + global=class label must reproduce measurements.anova exactly."""
        tokens, planted_global = make_planted_tokens(seed=1)
        N, T, D = tokens.shape
        x = tokens.to(dtype=torch.float64)
        image_means = x.mean(dim=1)
        within_per_image = (x - image_means.unsqueeze(1)).pow(2).sum(dim=(1, 2))
        records = LocalClusterRecords(
            image_ids=torch.arange(N, dtype=torch.int64),
            sizes=torch.full((N,), T, dtype=torch.int64),
            centroids=image_means,
            within_ss=within_per_image,
            tokens_per_image=T,
        )
        nested = compute_nested_anova(records, planted_global)
        legacy = compute_weighted_token_anova(x, planted_global)
        for new, old in [
            (nested.within_local, legacy.within_seq),
            (nested.between_local_in_global, legacy.within_class),
            (nested.between_global, legacy.between_class),
            (nested.total, legacy.total),
        ]:
            self.assertAlmostEqual(new, old, places=12)

    def test_assign_global_matches_brute_force(self) -> None:
        torch.manual_seed(0)
        records = LocalClusterRecords(
            image_ids=torch.arange(7, dtype=torch.int64),
            sizes=torch.ones(7, dtype=torch.int64),
            centroids=torch.randn(7, 5, dtype=torch.float64),
            within_ss=torch.zeros(7, dtype=torch.float64),
            tokens_per_image=1,
        )
        centers = torch.randn(3, 5, dtype=torch.float64)
        labels = assign_global(records, centers)
        brute = torch.cdist(records.centroids, centers).argmin(dim=1).to(torch.int64)
        self.assertTrue(torch.equal(labels, brute))

    def test_l2_normalize_unit_norm(self) -> None:
        x = torch.randn(4, 6, 8, dtype=torch.float64)
        x = l2_normalize_tokens(x)
        norms = x.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-12))


if __name__ == "__main__":
    unittest.main()
