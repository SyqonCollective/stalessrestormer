"""Data loading utilities for Restormer starless training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


def _list_images(folder: Path, valid_extensions: Sequence[str]) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in valid_extensions]
    files.sort()
    return files


def _random_crop_coords(width: int, height: int, crop_size: int) -> Tuple[int, int]:
    if width <= crop_size or height <= crop_size:
        return 0, 0
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    return left, top


class TileDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: int,
        augment: bool = True,
        random_crop: bool = True,
        valid_extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.random_crop = random_crop
        self.valid_extensions = valid_extensions

        input_dir = self.root / "input"
        target_dir = self.root / "target"

        inputs = _list_images(input_dir, valid_extensions)
        targets = _list_images(target_dir, valid_extensions)

        if not inputs or not targets:
            raise ValueError(f"Could not find tiles in {input_dir} / {target_dir}")

        input_map = {p.name: p for p in inputs}
        target_map = {p.name: p for p in targets}
        self.shared = sorted(set(input_map).intersection(target_map))
        if not self.shared:
            raise ValueError("No overlapping tiles between input and target")

        self.inputs = [input_map[name] for name in self.shared]
        self.targets = [target_map[name] for name in self.shared]

    def __len__(self) -> int:
        return len(self.inputs)

    def _open_pair(self, index: int) -> Tuple[Image.Image, Image.Image]:
        input_image = Image.open(self.inputs[index]).convert("RGB")
        target_image = Image.open(self.targets[index]).convert("RGB")
        return input_image, target_image

    def _apply_crop(self, input_image: Image.Image, target_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        width, height = input_image.size
        if self.random_crop:
            left, top = _random_crop_coords(width, height, self.image_size)
        else:
            left = max((width - self.image_size) // 2, 0)
            top = max((height - self.image_size) // 2, 0)
        box = (left, top, left + self.image_size, top + self.image_size)
        return input_image.crop(box), target_image.crop(box)

    def _augment(self, input_image: Image.Image, target_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            input_image = ImageOps.mirror(input_image)
            target_image = ImageOps.mirror(target_image)
        if random.random() < 0.5:
            input_image = ImageOps.flip(input_image)
            target_image = ImageOps.flip(target_image)
        if random.random() < 0.5:
            angle = random.randint(0, 3) * 90
            input_image = input_image.rotate(angle, expand=True)
            target_image = target_image.rotate(angle, expand=True)
        return input_image, target_image

    def __getitem__(self, index: int) -> dict:
        input_image, target_image = self._open_pair(index)
        if self.augment:
            input_image, target_image = self._augment(input_image, target_image)
        input_image, target_image = self._apply_crop(input_image, target_image)

        input_tensor = F.to_tensor(input_image) * 2 - 1
        target_tensor = F.to_tensor(target_image) * 2 - 1
        return {
            "input": input_tensor,
            "target": target_tensor,
            "name": self.inputs[index].name,
        }


def build_dataloader(
    root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    random_crop: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = TileDataset(
        root=root,
        image_size=image_size,
        augment=augment,
        random_crop=random_crop,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
