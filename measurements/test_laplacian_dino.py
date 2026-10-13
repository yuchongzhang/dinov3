from __future__ import annotations

import unittest

import torch

from measurements.laplacian_dino import (
    compute_laplacian_spectrum,
    l2_normalize_token_embeddings,
    pairwise_cosine_logits,
    patch_eigenvectors_for_render,
    random_walk_laplacian_from_logits,
    random_walk_transition_from_logits,
    rgb_overlay_from_eigenvectors,
)


class LaplacianDinoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.patch_tokens = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.1],
                [0.0, 0.1, 1.0],
            ]
        )
        self.cls_token = torch.tensor([[0.2, 0.3, 0.9]])

    def test_row_normalized_weights_match_softmax_identity(self) -> None:
        normalized = l2_normalize_token_embeddings(self.patch_tokens)
        logits = pairwise_cosine_logits(normalized, temperature=0.2)

        actual = random_walk_transition_from_logits(logits)
        expected = torch.softmax(logits, dim=1)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(dim=1), torch.ones(actual.shape[0], dtype=actual.dtype))

    def test_random_walk_eigenvectors_satisfy_eigen_equation(self) -> None:
        normalized = l2_normalize_token_embeddings(self.patch_tokens)
        logits = pairwise_cosine_logits(normalized, temperature=0.5)
        laplacian = random_walk_laplacian_from_logits(logits)
        spectrum = compute_laplacian_spectrum(
            self.patch_tokens,
            temperature=0.5,
            n_eigenvectors=4,
        )

        residual = laplacian @ spectrum.eigenvectors - spectrum.eigenvectors * spectrum.eigenvalues.unsqueeze(0)

        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-10, rtol=1e-8)

    def test_eigenvalues_are_smallest_sorted_values(self) -> None:
        spectrum = compute_laplacian_spectrum(
            self.patch_tokens,
            temperature=0.4,
            n_eigenvectors=4,
        )

        self.assertTrue(torch.all(spectrum.eigenvalues[1:] >= spectrum.eigenvalues[:-1]))
        self.assertAlmostEqual(float(spectrum.eigenvalues[0]), 0.0, places=10)

    def test_full_spectrum_keeps_all_eigenvalues_when_visualizing_fewer_vectors(self) -> None:
        spectrum = compute_laplacian_spectrum(
            torch.cat((self.cls_token, self.patch_tokens), dim=0),
            temperature=0.3,
            n_eigenvectors=4,
        )

        self.assertEqual(spectrum.eigenvalues.numel(), 4)
        self.assertEqual(spectrum.all_eigenvalues.numel(), 5)
        self.assertTrue(torch.all(spectrum.all_eigenvalues[1:] >= spectrum.all_eigenvalues[:-1]))
        torch.testing.assert_close(spectrum.eigenvalues, spectrum.all_eigenvalues[:4])

    def test_cls_inclusion_changes_graph_size_but_patch_rendering_stays_spatial(self) -> None:
        patch_spectrum = compute_laplacian_spectrum(
            self.patch_tokens,
            temperature=0.3,
            n_eigenvectors=4,
        )
        with_cls_spectrum = compute_laplacian_spectrum(
            torch.cat((self.cls_token, self.patch_tokens), dim=0),
            temperature=0.3,
            n_eigenvectors=4,
        )

        self.assertEqual(patch_spectrum.eigenvectors.shape[0], 4)
        self.assertEqual(with_cls_spectrum.eigenvectors.shape[0], 5)

        patch_only_for_render = patch_eigenvectors_for_render(
            patch_spectrum.eigenvectors,
            include_cls=False,
            expected_patches=4,
        )
        with_cls_for_render = patch_eigenvectors_for_render(
            with_cls_spectrum.eigenvectors,
            include_cls=True,
            expected_patches=4,
        )

        self.assertEqual(patch_only_for_render.shape, (4, 4))
        self.assertEqual(with_cls_for_render.shape, (4, 4))
        self.assertEqual(rgb_overlay_from_eigenvectors(with_cls_for_render).shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
