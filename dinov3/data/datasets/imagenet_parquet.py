# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import glob
import logging
import mmap
import os
from typing import Callable, Optional, Union

import numpy as np
import torch.distributed as dist

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset
from .image_net import _Split

logger = logging.getLogger("dinov3")
_Target = int

# Map _Split enum values to HuggingFace parquet file prefixes
_SPLIT_TO_PREFIX = {
    _Split.TRAIN: "train",
    _Split.VAL: "validation",
    _Split.TEST: "test",
}

_CACHE_IMAGES = "images.bin"
_CACHE_OFFSETS = "offsets.npy"
_CACHE_LABELS = "labels.npy"


def _build_cache(parquet_dir: str, split: _Split, cache_dir: str) -> None:
    """Convert parquet files to a flat binary cache for O(1) random access.

    Creates three files in cache_dir:
      images.bin  – concatenated raw JPEG bytes
      offsets.npy – int64 byte-offset array (length N+1)
      labels.npy  – int64 label array (length N)
    """
    import pyarrow.parquet as pq

    prefix = _SPLIT_TO_PREFIX[split]
    files = sorted(glob.glob(os.path.join(parquet_dir, f"{prefix}-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files for split '{split.value}' (prefix='{prefix}') in {parquet_dir}"
        )

    os.makedirs(cache_dir, exist_ok=True)
    tmp_bin = os.path.join(cache_dir, _CACHE_IMAGES + ".tmp")

    offsets = [0]
    labels_parts = []

    with open(tmp_bin, "wb") as f:
        for file_idx, pq_path in enumerate(files):
            logger.info(
                f"  [{file_idx + 1}/{len(files)}] {os.path.basename(pq_path)}"
            )
            table = pq.read_table(pq_path, memory_map=True)
            labels_parts.append(table.column("label").to_numpy())

            for chunk in table.column("image").chunks:
                bytes_array = chunk.field("bytes")
                for i in range(len(bytes_array)):
                    img_bytes = bytes_array[i].as_py()
                    f.write(img_bytes)
                    offsets.append(offsets[-1] + len(img_bytes))
        f.flush()
        os.fsync(f.fileno())

    offsets_arr = np.array(offsets, dtype=np.int64)
    labels_arr = np.concatenate(labels_parts)
    np.save(os.path.join(cache_dir, _CACHE_OFFSETS), offsets_arr)
    np.save(os.path.join(cache_dir, _CACHE_LABELS), labels_arr)

    # Atomic rename signals the cache is complete
    os.rename(tmp_bin, os.path.join(cache_dir, _CACHE_IMAGES))
    logger.info(
        f"Cache ready: {len(labels_arr):,d} images, "
        f"{offsets_arr[-1] / 1e9:.1f} GB in {cache_dir}"
    )


class ImageNetParquet(ExtendedVisionDataset):
    """ImageNet dataset backed by HuggingFace-format parquet files.

    On first use, converts parquet data into a flat binary cache
    (images.bin + offsets.npy + labels.npy) for O(1) memory-mapped
    random access.  Subsequent runs load the cache directly.

    Parquet files must have columns:
      - image: struct<bytes: binary, path: string>
      - label: int64

    Usage in dataset string:
      ImageNetParquet:split=TRAIN:root=/path/to/data
      ImageNetParquet:split=TRAIN:root=/path/to/data:extra=/path/to/cache
    """

    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "ImageNetParquet.Split",
        root: str,
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self._split = split
        self._cache_dir = extra or os.path.join(root, f".cache_{split.value}")

        # Build cache if it doesn't exist yet (only rank 0 builds, others wait)
        cache_bin = os.path.join(self._cache_dir, _CACHE_IMAGES)
        is_distributed = dist.is_available() and dist.is_initialized()
        is_main = not is_distributed or dist.get_rank() == 0
        if not os.path.exists(cache_bin) and is_main:
            logger.info(
                f"Binary cache not found at {self._cache_dir}. "
                f"Building from parquet (one-time)..."
            )
            _build_cache(root, split, self._cache_dir)
        if is_distributed:
            dist.barrier()

        # Load cache: memory-mapped for O(1) random access
        self._offsets = np.load(
            os.path.join(self._cache_dir, _CACHE_OFFSETS), mmap_mode="r"
        )
        self._labels = np.load(
            os.path.join(self._cache_dir, _CACHE_LABELS), mmap_mode="r"
        )
        self._total_rows = len(self._labels)

        self._images_fd = open(cache_bin, "rb")
        self._images_mmap = mmap.mmap(
            self._images_fd.fileno(), 0, access=mmap.ACCESS_READ
        )

        logger.info(
            f"ImageNetParquet [{split.value}]: {self._total_rows:,d} samples "
            f"from cache {self._cache_dir}"
        )

    @property
    def split(self) -> "ImageNetParquet.Split":
        return self._split

    def get_image_data(self, index: int) -> bytes:
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        return bytes(self._images_mmap[start:end])

    def get_target(self, index: int) -> Optional[_Target]:
        return None if self._split == _Split.TEST else int(self._labels[index])

    def get_targets(self) -> Optional[np.ndarray]:
        return None if self._split == _Split.TEST else self._labels

    def __len__(self) -> int:
        return self._total_rows
