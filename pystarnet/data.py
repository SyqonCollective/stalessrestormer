"""Data utilities for PyStarNet."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from .configs import DatasetConfig, TrainerConfig


def _list_images(folder: Path, valid_extensions: Sequence[str]) -> List[Path]:
    folder = folder.expanduser()
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in valid_extensions]
    files.sort()
    return files


def _random_crop_coords(width: int, height: int, crop_size: int) -> Tuple[int, int]:
    if width == crop_size and height == crop_size:
        return 0, 0
    if width < crop_size or height < crop_size:
        raise ValueError("Tile is smaller than crop size")
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    return left, top


class TileDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        role: str = "train",
    ) -> None:
        self.root = Path(config.root)
        self.role = role
        self.crop_size = config.image_size
        self.random_crop = config.random_crop and role == "train"
        self.augment = config.augment and role == "train"
        self.valid_extensions = config.valid_extensions

        input_dir = self.root / "input"
        target_dir = self.root / "target"
        self.inputs = _list_images(input_dir, self.valid_extensions)
        self.targets = _list_images(target_dir, self.valid_extensions)

        if len(self.inputs) == 0:
            raise ValueError(f"No tiles found in {input_dir}")
        if self.inputs != self.targets:
            raise ValueError("Input and target tile names do not match")

    def __len__(self) -> int:
        return len(self.inputs)

    def _open_pair(self, index: int) -> Tuple[Image.Image, Image.Image]:
        input_image = Image.open(self.inputs[index]).convert("RGB")
        target_image = Image.open(self.targets[index]).convert("RGB")
        return input_image, target_image

    def _apply_crop(self, input_image: Image.Image, target_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        width, height = input_image.size
        if self.random_crop:
            left, top = _random_crop_coords(width, height, self.crop_size)
        else:
            left = max((width - self.crop_size) // 2, 0)
            top = max((height - self.crop_size) // 2, 0)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        return input_image.crop(box), target_image.crop(box)

    def _augment(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=(2,))
            target_tensor = torch.flip(target_tensor, dims=(2,))
        if random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=(1,))
            target_tensor = torch.flip(target_tensor, dims=(1,))
        if random.random() < 0.5:
            input_tensor = input_tensor.transpose(1, 2)
            target_tensor = target_tensor.transpose(1, 2)
        return input_tensor, target_tensor

    def __getitem__(self, index: int) -> dict:
        input_image, target_image = self._open_pair(index)
        input_image, target_image = self._apply_crop(input_image, target_image)

        input_tensor = F.to_tensor(input_image)
        target_tensor = F.to_tensor(target_image)

        if self.augment:
            input_tensor, target_tensor = self._augment(input_tensor, target_tensor)

        input_tensor = input_tensor * 2.0 - 1.0
        target_tensor = target_tensor * 2.0 - 1.0

        return {
            "input": input_tensor,
            "target": target_tensor,
            "name": self.inputs[index].name,
        }


def build_dataloader(
    dataset_config: DatasetConfig,
    trainer_config: TrainerConfig,
    role: str,
) -> DataLoader:
    dataset = TileDataset(dataset_config, role=role)
    shuffle = role == "train"
    return DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=shuffle,
        num_workers=trainer_config.num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
