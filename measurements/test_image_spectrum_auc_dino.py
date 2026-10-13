from __future__ import annotations

import math
import unittest

import torch

from measurements.image_spectrum_auc_dino import (
    batch_singular_value_energy_auc,
    compute_image_spectrum_auc_stats,
    cumulative_energy_curve,
    normalized_squared_singular_value_energy,
    singular_value_energy_auc,
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


def auc_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return singular_value_energy_auc(torch.linalg.svdvals(tokens))


class ImageSpectrumAUCDinoTest(unittest.TestCase):
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

    def test_energy_curve_and_auc_match_definition(self) -> None:
        singular_values = torch.tensor([4.0, 3.0])

        energy = normalized_squared_singular_value_energy(singular_values)
        cumulative = cumulative_energy_curve(singular_values)
        auc = singular_value_energy_auc(singular_values)

        torch.testing.assert_close(energy, torch.tensor([0.64, 0.36]))
        torch.testing.assert_close(cumulative, torch.tensor([0.64, 1.0]))
        self.assertAlmostEqual(float(auc), 0.82, places=7)

    def test_batch_singular_value_energy_auc_matches_per_image_svd(self) -> None:
        tokens = torch.tensor(
            [
                [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]],
                [[2.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        )

        actual = batch_singular_value_energy_auc(tokens)
        expected = torch.stack([auc_from_tokens(image_tokens) for image_tokens in tokens])

        torch.testing.assert_close(actual, expected)

    def test_compute_image_spectrum_auc_stats_uses_patch_tokens_and_caps_images(self) -> None:
        stats = compute_image_spectrum_auc_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            exclude_cls=True,
        )

        per_image = torch.stack([auc_from_tokens(image_tokens) for image_tokens in self.tokens[:3, 1:, :]]).double()
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

    def test_compute_image_spectrum_auc_stats_can_include_cls(self) -> None:
        stats = compute_image_spectrum_auc_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=2,
            exclude_cls=False,
        )

        per_image = torch.stack([auc_from_tokens(image_tokens) for image_tokens in self.tokens[:2]]).double()

        self.assertEqual(stats.num_images, 2)
        self.assertEqual(stats.tokens_per_image, 3)
        self.assertEqual(stats.embed_dim, 2)
        self.assertEqual(stats.spectrum_length, 2)
        self.assertAlmostEqual(stats.mean, float(per_image.mean()), places=6)


if __name__ == "__main__":
    unittest.main()
