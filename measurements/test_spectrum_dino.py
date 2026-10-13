from __future__ import annotations

import unittest

import torch

from measurements.spectrum_dino import (
    compute_dataset_spectrum,
    compute_global_token_mean,
    cutoff_length_for_spectrum,
    extract_last_layer_tokens,
    gram_eigenvalues,
    parse_args,
    spectrum_for_plot,
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


class SpectrumDinoTest(unittest.TestCase):
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

    def test_extract_last_layer_tokens_includes_cls_by_default(self) -> None:
        images = torch.tensor([[0.0], [1.0]])
        actual = extract_last_layer_tokens(
            self.model,
            images,
            autocast_dtype=torch.float32,
            exclude_cls=False,
        )
        torch.testing.assert_close(actual, self.tokens[:2])

    def test_extract_last_layer_tokens_can_exclude_cls(self) -> None:
        images = torch.tensor([[0.0], [1.0]])
        actual = extract_last_layer_tokens(
            self.model,
            images,
            autocast_dtype=torch.float32,
            exclude_cls=True,
        )
        torch.testing.assert_close(actual, self.tokens[:2, 1:, :])

    def test_parse_args_defaults_to_patch_tokens(self) -> None:
        args = parse_args([])

        self.assertTrue(args.exclude_cls)

    def test_compute_dataset_spectrum_accumulates_patch_gram_and_caps_images(self) -> None:
        spectrum, num_images, tokens_per_image, embed_dim = compute_dataset_spectrum(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            exclude_cls=True,
        )

        flat_tokens = self.tokens[:3, 1:, :].reshape(-1, self.tokens.shape[-1])
        expected = gram_eigenvalues(flat_tokens.T.double() @ flat_tokens.double())

        self.assertEqual(num_images, 3)
        self.assertEqual(tokens_per_image, 2)
        self.assertEqual(embed_dim, 2)
        torch.testing.assert_close(spectrum, expected)

    def test_compute_dataset_spectrum_can_include_cls(self) -> None:
        spectrum, num_images, tokens_per_image, embed_dim = compute_dataset_spectrum(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            exclude_cls=False,
        )

        flat_tokens = self.tokens[:3].reshape(-1, self.tokens.shape[-1])
        expected = gram_eigenvalues(flat_tokens.T.double() @ flat_tokens.double())

        self.assertEqual(num_images, 3)
        self.assertEqual(tokens_per_image, 3)
        self.assertEqual(embed_dim, 2)
        torch.testing.assert_close(spectrum, expected)

    def test_compute_dataset_spectrum_excludes_cls_for_full_dataset(self) -> None:
        spectrum, num_images, tokens_per_image, embed_dim = compute_dataset_spectrum(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=-1,
            exclude_cls=True,
        )

        flat_tokens = self.tokens[:, 1:, :].reshape(-1, self.tokens.shape[-1])
        expected = gram_eigenvalues(flat_tokens.T.double() @ flat_tokens.double())

        self.assertEqual(num_images, 4)
        self.assertEqual(tokens_per_image, 2)
        self.assertEqual(embed_dim, 2)
        torch.testing.assert_close(spectrum, expected)

    def test_compute_global_token_mean_uses_selected_tokens_and_caps_images(self) -> None:
        mean, num_images, tokens_per_image, embed_dim = compute_global_token_mean(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            exclude_cls=True,
        )

        flat_tokens = self.tokens[:3, 1:, :].reshape(-1, self.tokens.shape[-1])
        expected = flat_tokens.mean(dim=0)

        self.assertEqual(num_images, 3)
        self.assertEqual(tokens_per_image, 2)
        self.assertEqual(embed_dim, 2)
        torch.testing.assert_close(mean, expected)

    def test_compute_dataset_spectrum_can_subtract_global_token_mean(self) -> None:
        global_mean, mean_images, mean_tokens_per_image, mean_embed_dim = compute_global_token_mean(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            exclude_cls=True,
        )
        spectrum, num_images, tokens_per_image, embed_dim = compute_dataset_spectrum(
            model=self.model,
            data_loader=make_loader([2, 2]),
            device=torch.device("cpu"),
            autocast_dtype=torch.float32,
            log_every=0,
            n_images=3,
            exclude_cls=True,
            global_mean=global_mean,
        )

        flat_tokens = self.tokens[:3, 1:, :].reshape(-1, self.tokens.shape[-1])
        centered = flat_tokens - flat_tokens.mean(dim=0, keepdim=True)
        expected = gram_eigenvalues(centered.T.double() @ centered.double())

        self.assertEqual(mean_images, 3)
        self.assertEqual(mean_tokens_per_image, 2)
        self.assertEqual(mean_embed_dim, 2)
        self.assertEqual(num_images, 3)
        self.assertEqual(tokens_per_image, 2)
        self.assertEqual(embed_dim, 2)
        torch.testing.assert_close(spectrum, expected)

    def test_spectrum_for_plot_keeps_only_top_k_by_index(self) -> None:
        spectrum = torch.tensor([5.0, 4.0, 3.0, 2.0])

        torch.testing.assert_close(spectrum_for_plot(spectrum, -1), spectrum)
        torch.testing.assert_close(spectrum_for_plot(spectrum, 2), torch.tensor([5.0, 4.0]))
        torch.testing.assert_close(spectrum_for_plot(spectrum, 10), spectrum)

    def test_cutoff_length_stops_before_largest_log_drop_and_handles_degenerate_spectra(self) -> None:
        self.assertEqual(cutoff_length_for_spectrum(torch.tensor([1000.0, 100.0, 10.0, 0.1, 0.09])), 3)
        self.assertEqual(cutoff_length_for_spectrum(torch.tensor([1.0, 0.0, 0.0])), 3)
        self.assertEqual(cutoff_length_for_spectrum(torch.tensor([0.0, 0.0])), 2)


if __name__ == "__main__":
    unittest.main()
