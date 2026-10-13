from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from measurements.compute_block_cossim import compute_cosine_similarity
from measurements.dino_measurement_utils import build_balanced_batch, concatenate_layer_tokens, extract_block_tokens
from measurements.pca_dino import compute_projected_image, load_foreground_mask
from measurements.plot_block_pca import compute_pca_projection


class TinyTargetDataset(torch.utils.data.Dataset):
    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.tensor([float(index)]), torch.tensor(self.targets[index])


class TinyDinoLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_blocks = 2
        self.norm = torch.nn.LayerNorm(1)

    def get_intermediate_layers(self, x, *, n, return_class_token, return_extra_tokens, norm):
        self.assert_contract(return_class_token, return_extra_tokens, norm)
        outputs = []
        for layer_idx in n:
            patch_tokens = torch.full((x.shape[0], 2, 1), float(10 + layer_idx))
            class_token = torch.full((x.shape[0], 1), float(layer_idx))
            extra_tokens = torch.full((x.shape[0], 1, 1), float(5 + layer_idx))
            outputs.append((patch_tokens, class_token, extra_tokens))
        return tuple(outputs)

    @staticmethod
    def assert_contract(return_class_token, return_extra_tokens, norm) -> None:
        if not return_class_token or not return_extra_tokens or norm:
            raise AssertionError("Unexpected get_intermediate_layers arguments.")


class TinyNoExtraDinoLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_blocks = 1
        self.norm = torch.nn.LayerNorm(1)

    def get_intermediate_layers(self, x, *, n, return_class_token, norm):
        if not return_class_token or norm:
            raise AssertionError("Unexpected get_intermediate_layers arguments.")
        outputs = []
        for layer_idx in n:
            patch_tokens = torch.full((x.shape[0], 2, 1), float(20 + layer_idx))
            class_token = torch.full((x.shape[0], 1), float(layer_idx))
            outputs.append((patch_tokens, class_token))
        return tuple(outputs)


class BlockMetricsTest(unittest.TestCase):
    def test_pca_projection_shape_and_centering(self) -> None:
        tokens = torch.tensor(
            [
                [[1.0, 0.0, 2.0], [2.0, 1.0, 0.0]],
                [[3.0, 2.0, 1.0], [4.0, 3.0, 2.0]],
            ]
        )
        projection = compute_pca_projection(tokens)

        self.assertEqual(tuple(projection.shape), (2, 2, 2))
        self.assertTrue(torch.allclose(projection.mean(dim=(0, 1)), torch.zeros(2), atol=1e-6))

    def test_dino_pca_projection_can_use_all_patches(self) -> None:
        patch_features = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        projection = compute_projected_image(
            patch_features,
            h_patches=2,
            w_patches=2,
            foreground_mask=None,
        )

        self.assertEqual(tuple(projection.shape), (3, 2, 2))
        self.assertTrue(bool((projection > 0).any()))

    def test_dino_pca_projection_applies_foreground_mask(self) -> None:
        patch_features = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        foreground_mask = torch.tensor([[True, True], [True, False]])

        projection = compute_projected_image(
            patch_features,
            h_patches=2,
            w_patches=2,
            foreground_mask=foreground_mask,
        )

        self.assertTrue(torch.equal(projection[:, 1, 1], torch.zeros(3)))

    def test_load_foreground_mask_thresholds_saved_score_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "fg_score.pt"
            torch.save(torch.tensor([0.2, 0.7, 0.5, 0.9]), mask_path)

            mask = load_foreground_mask(mask_path, 2, 2, foreground_threshold=0.5)

        self.assertEqual(mask.tolist(), [[False, True], [False, True]])

    def test_cosine_similarity_averages_off_diagonal_entries(self) -> None:
        tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
        actual = compute_cosine_similarity(tokens)

        self.assertAlmostEqual(actual, -1.0 / 3.0, places=7)

    def test_concatenate_layer_tokens_uses_dino_token_order(self) -> None:
        patch_tokens = torch.tensor([[[10.0], [11.0]]])
        class_token = torch.tensor([[1.0]])
        extra_tokens = torch.tensor([[[2.0], [3.0]]])

        tokens = concatenate_layer_tokens((patch_tokens, class_token, extra_tokens))

        self.assertEqual(tokens.squeeze(-1).tolist(), [[1.0, 2.0, 3.0, 10.0, 11.0]])

    def test_concatenate_layer_tokens_flattens_spatial_patch_tokens(self) -> None:
        patch_tokens = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        class_token = torch.tensor([[0.0]])

        tokens = concatenate_layer_tokens((patch_tokens, class_token))

        self.assertEqual(tokens.squeeze(-1).tolist(), [[0.0, 1.0, 2.0, 3.0, 4.0]])

    def test_extract_block_tokens_returns_blocks_and_final_norm(self) -> None:
        model = TinyDinoLikeModel()
        images = torch.zeros(2, 3, 4, 4)

        layer_tokens = extract_block_tokens(model, images, autocast_dtype=torch.float32)

        self.assertEqual(list(layer_tokens.keys()), ["block_0", "block_1", "norm_final"])
        self.assertEqual(tuple(layer_tokens["block_0"].shape), (2, 4, 1))
        with torch.no_grad():
            expected = model.norm(layer_tokens["block_1"].clone())
        self.assertTrue(torch.allclose(layer_tokens["norm_final"], expected))

    def test_extract_block_tokens_supports_no_extra_token_signature(self) -> None:
        model = TinyNoExtraDinoLikeModel()
        images = torch.zeros(2, 3, 4, 4)

        layer_tokens = extract_block_tokens(model, images, autocast_dtype=torch.float32)

        self.assertEqual(list(layer_tokens.keys()), ["block_0", "norm_final"])
        self.assertEqual(tuple(layer_tokens["block_0"].shape), (2, 3, 1))

    def test_build_balanced_batch_samples_requested_classes(self) -> None:
        dataset = TinyTargetDataset([0, 0, 1, 1, 1, 2])

        _, labels = build_balanced_batch(
            dataset,
            selected_classes=[0, 1],
            samples_per_class=2,
            seed=7,
            batch_size=2,
            num_workers=0,
        )

        self.assertEqual(labels.numel(), 4)
        self.assertEqual(int((labels == 0).sum().item()), 2)
        self.assertEqual(int((labels == 1).sum().item()), 2)
        self.assertEqual(set(labels.tolist()), {0, 1})

    def test_build_balanced_batch_rejects_underfilled_class(self) -> None:
        dataset = TinyTargetDataset([0, 1, 1])

        with self.assertRaises(ValueError):
            build_balanced_batch(
                dataset,
                selected_classes=[0],
                samples_per_class=2,
                seed=7,
                batch_size=2,
                num_workers=0,
            )


if __name__ == "__main__":
    unittest.main()
