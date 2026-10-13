from __future__ import annotations

import unittest

import torch

from measurements.image_spectrum_dino import (
    center_tokens_per_image,
    compute_image_spectrum_stats,
    image_gram_eigenvalues,
    parse_args,
    spectrum_length_for_plot,
)
from measurements.spectrum_dino import gram_eigenvalues


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


class ImageSpectrumDinoTest(unittest.TestCase):
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

    def test_spectrum_length_uses_top_k_or_full_embed_dim(self) -> None:
        self.assertEqual(spectrum_length_for_plot(embed_dim=8, top_k_eig=-1), 8)
        self.assertEqual(spectrum_length_for_plot(embed_dim=8, top_k_eig=3), 3)
        self.assertEqual(spectrum_length_for_plot(embed_dim=8, top_k_eig=12), 8)

    def test_image_gram_eigenvalues_matches_explicit_gram_and_pads_zeros(self) -> None:
        tokens = torch.tensor(
            [
                [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]],
                [[2.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        )

        actual = image_gram_eigenvalues(tokens, spectrum_length=3)
        expected = torch.stack(
            [gram_eigenvalues(image_tokens.T.double() @ image_tokens.double()) for image_tokens in tokens]
        )

        torch.testing.assert_close(actual, expected)

    def test_parse_args_defaults_to_patch_tokens(self) -> None:
        args = parse_args([])

        self.assertTrue(args.exclude_cls)

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

    def test_image_gram_eigenvalues_can_center_each_image_before_svd(self) -> None:
        tokens = torch.tensor(
            [
                [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]],
                [[2.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        )

        actual = image_gram_eigenvalues(tokens, spectrum_length=3, center_per_image=True)
        expected = torch.stack(
            [
                gram_eigenvalues(centered.T.double() @ centered.double())
                for centered in tokens - tokens.mean(dim=1, keepdim=True)
            ]
        )

        torch.testing.assert_close(actual, expected)

    def test_compute_image_spectrum_stats_defaults_to_patch_tokens_and_caps_images(self) -> None:
        stats = compute_image_spectrum_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            top_k_eig=-1,
        )

        per_image = torch.stack(
            [
                gram_eigenvalues(image_tokens.T.double() @ image_tokens.double())
                for image_tokens in self.tokens[:3, 1:, :]
            ]
        )
        expected_mean = per_image.mean(dim=0)
        expected_std = per_image.std(dim=0, unbiased=False)

        self.assertEqual(stats.num_images, 3)
        self.assertEqual(stats.tokens_per_image, 2)
        self.assertEqual(stats.embed_dim, 2)
        self.assertEqual(stats.spectrum_length, 2)
        torch.testing.assert_close(stats.mean, expected_mean)
        torch.testing.assert_close(stats.std, expected_std)

    def test_compute_image_spectrum_stats_can_include_cls(self) -> None:
        stats = compute_image_spectrum_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            top_k_eig=-1,
            exclude_cls=False,
        )

        per_image = torch.stack(
            [gram_eigenvalues(image_tokens.T.double() @ image_tokens.double()) for image_tokens in self.tokens[:3]]
        )

        self.assertEqual(stats.num_images, 3)
        self.assertEqual(stats.tokens_per_image, 3)
        self.assertEqual(stats.embed_dim, 2)
        self.assertEqual(stats.spectrum_length, 2)
        torch.testing.assert_close(stats.mean, per_image.mean(dim=0))
        torch.testing.assert_close(stats.std, per_image.std(dim=0, unbiased=False))

    def test_compute_image_spectrum_stats_can_center_patch_tokens_per_image(self) -> None:
        stats = compute_image_spectrum_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            top_k_eig=-1,
            exclude_cls=True,
            center_per_image=True,
        )

        patch_tokens = self.tokens[:3, 1:, :]
        centered = patch_tokens - patch_tokens.mean(dim=1, keepdim=True)
        per_image = torch.stack(
            [gram_eigenvalues(image_tokens.T.double() @ image_tokens.double()) for image_tokens in centered]
        )

        self.assertEqual(stats.num_images, 3)
        self.assertEqual(stats.tokens_per_image, 2)
        self.assertEqual(stats.embed_dim, 2)
        self.assertEqual(stats.spectrum_length, 2)
        torch.testing.assert_close(stats.mean, per_image.mean(dim=0))
        torch.testing.assert_close(stats.std, per_image.std(dim=0, unbiased=False))

    def test_compute_image_spectrum_stats_excludes_cls_and_keeps_top_k(self) -> None:
        stats = compute_image_spectrum_stats(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=-1,
            top_k_eig=1,
            exclude_cls=True,
        )

        per_image = torch.stack(
            [
                gram_eigenvalues(image_tokens.T.double() @ image_tokens.double())[:1]
                for image_tokens in self.tokens[:, 1:, :]
            ]
        )

        self.assertEqual(stats.num_images, 4)
        self.assertEqual(stats.tokens_per_image, 2)
        self.assertEqual(stats.embed_dim, 2)
        self.assertEqual(stats.spectrum_length, 1)
        torch.testing.assert_close(stats.mean, per_image.mean(dim=0))
        torch.testing.assert_close(stats.std, per_image.std(dim=0, unbiased=False))


if __name__ == "__main__":
    unittest.main()
