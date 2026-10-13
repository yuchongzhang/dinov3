from __future__ import annotations

import math
import unittest

import torch

from measurements.image_spectrum_kalpha_dino import (
    batch_singular_value_energy_k_alpha,
    center_tokens_per_image,
    compute_image_spectrum_k_alpha_stats,
    cumulative_energy_curve,
    normalized_squared_singular_value_energy,
    singular_value_energy_k_alpha,
)


class ToyTokenModel:
    def __init__(self, tokens: torch.Tensor) -> None:
        self.tokens = tokens

    def get_intermediate_layers(self, images, n, return_class_token, norm):
        indices = images[:, 0].long()
        tokens = self.tokens[indices]
        return [(tokens[:, 1:, :], tokens[:, 0, :])]


def make_loader(batch_sizes: list[int]):
    start = 0
    batches = []
    for batch_size in batch_sizes:
        stop = start + batch_size
        samples = torch.arange(start, stop, dtype=torch.float32).unsqueeze(1)
        targets = torch.zeros(batch_size, dtype=torch.long)
        batches.append((samples, targets))
        start = stop
    return batches


def k_alpha_from_tokens(tokens: torch.Tensor, *, alpha: float) -> torch.Tensor:
    return singular_value_energy_k_alpha(torch.linalg.svdvals(tokens), alpha=alpha)


class ImageSpectrumKAlphaDinoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = torch.tensor(
            [
                [[10.0, 1.0], [0.0, 1.0], [2.0, 0.0]],
                [[20.0, 2.0], [1.0, 1.0], [3.0, 1.0]],
                [[30.0, 3.0], [4.0, 0.0], [0.0, 2.0]],
                [[40.0, 4.0], [1.0, 3.0], [2.0, 2.0]],
            ]
        )
        self.model = ToyTokenModel(self.tokens)

    def test_energy_curve_and_k_alpha_match_definition(self) -> None:
        singular_values = torch.tensor([4.0, 3.0])

        energy = normalized_squared_singular_value_energy(singular_values)
        cumulative = cumulative_energy_curve(singular_values)
        k_60 = singular_value_energy_k_alpha(singular_values, alpha=0.6)
        k_90 = singular_value_energy_k_alpha(singular_values, alpha=0.9)

        torch.testing.assert_close(energy, torch.tensor([0.64, 0.36]))
        torch.testing.assert_close(cumulative, torch.tensor([0.64, 1.0]))
        self.assertEqual(int(k_60), 1)
        self.assertEqual(int(k_90), 2)

    def test_batch_singular_value_energy_k_alpha_matches_per_image_svd(self) -> None:
        tokens = torch.tensor(
            [
                [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]],
                [[2.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        )

        actual = batch_singular_value_energy_k_alpha(tokens, alpha=0.85)
        expected = torch.stack([k_alpha_from_tokens(image_tokens, alpha=0.85) for image_tokens in tokens])

        torch.testing.assert_close(actual, expected)

    def test_center_tokens_per_image_subtracts_each_image_token_mean(self) -> None:
        tokens = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]],
                [[10.0, 0.0], [10.0, 2.0], [16.0, 4.0]],
            ]
        )

        centered = center_tokens_per_image(tokens)
        expected = tokens - tokens.mean(dim=1, keepdim=True)

        torch.testing.assert_close(centered, expected)
        torch.testing.assert_close(centered.mean(dim=1), torch.zeros(2, 2))

    def test_compute_image_spectrum_k_alpha_stats_uses_patch_tokens_and_caps_images(self) -> None:
        stats = compute_image_spectrum_k_alpha_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            alpha=0.8,
            exclude_cls=True,
        )

        per_image = torch.stack(
            [k_alpha_from_tokens(image_tokens, alpha=0.8) for image_tokens in self.tokens[:3, 1:, :]]
        ).double()
        expected_mean = float(per_image.mean())
        expected_std = float(per_image.std(unbiased=True))
        expected_standard_error = expected_std / math.sqrt(per_image.numel())

        self.assertEqual(stats.num_images, 3)
        self.assertEqual(stats.tokens_per_image, 2)
        self.assertEqual(stats.embed_dim, 2)
        self.assertEqual(stats.spectrum_length, 2)
        self.assertAlmostEqual(stats.mean, expected_mean, places=7)
        self.assertAlmostEqual(stats.std, expected_std, places=7)
        self.assertAlmostEqual(stats.standard_error, expected_standard_error, places=7)
        self.assertEqual(stats.min_k_alpha, int(per_image.min()))
        self.assertEqual(stats.max_k_alpha, int(per_image.max()))

    def test_compute_image_spectrum_k_alpha_stats_can_center_patch_tokens_per_image(self) -> None:
        stats = compute_image_spectrum_k_alpha_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            alpha=0.8,
            exclude_cls=True,
            center_per_image=True,
        )

        patch_tokens = self.tokens[:3, 1:, :]
        centered = patch_tokens - patch_tokens.mean(dim=1, keepdim=True)
        per_image = torch.stack(
            [k_alpha_from_tokens(image_tokens, alpha=0.8) for image_tokens in centered]
        ).double()

        self.assertEqual(stats.num_images, 3)
        self.assertEqual(stats.tokens_per_image, 2)
        self.assertEqual(stats.embed_dim, 2)
        self.assertEqual(stats.spectrum_length, 2)
        self.assertAlmostEqual(stats.mean, float(per_image.mean()), places=7)
        self.assertAlmostEqual(stats.std, float(per_image.std(unbiased=True)), places=7)

    def test_compute_image_spectrum_k_alpha_stats_can_include_cls(self) -> None:
        stats = compute_image_spectrum_k_alpha_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=2,
            alpha=0.8,
            exclude_cls=False,
        )

        per_image = torch.stack([k_alpha_from_tokens(image_tokens, alpha=0.8) for image_tokens in self.tokens[:2]])

        self.assertEqual(stats.num_images, 2)
        self.assertEqual(stats.tokens_per_image, 3)
        self.assertEqual(stats.embed_dim, 2)
        self.assertEqual(stats.spectrum_length, 2)
        self.assertAlmostEqual(stats.mean, float(per_image.double().mean()), places=7)


if __name__ == "__main__":
    unittest.main()
