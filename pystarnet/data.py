"""Data utilities for PyStarNet."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
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
        input_files = _list_images(input_dir, self.valid_extensions)
        target_files = _list_images(target_dir, self.valid_extensions)

        if not input_files:
            raise ValueError(f"No tiles found in {input_dir}")
        if not target_files:
            raise ValueError(f"No tiles found in {target_dir}")

        input_map = {p.name: p for p in input_files}
        target_map = {p.name: p for p in target_files}
        shared = sorted(set(input_map).intersection(target_map))

        missing_input = sorted(set(target_map).difference(input_map))
        missing_target = sorted(set(input_map).difference(target_map))

        if missing_input or missing_target:
            message = ["Tile mismatch detected; using intersection of names."]
            if missing_input:
                message.append(f"Missing in input/: {', '.join(missing_input[:5])}")
                if len(missing_input) > 5:
                    message.append(f"...and {len(missing_input) - 5} more")
            if missing_target:
                message.append(f"Missing in target/: {', '.join(missing_target[:5])}")
                if len(missing_target) > 5:
                    message.append(f"...and {len(missing_target) - 5} more")
            print("[PyStarNet] " + " ".join(message))

        if not shared:
            raise ValueError("No overlapping tile names between input/ and target/")

        self.inputs = [input_map[name] for name in shared]
        self.targets = [target_map[name] for name in shared]

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

    def _apply_starnet_augmentation(self, input_image: Image.Image, target_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.33:
            angle = random.randint(0, 359)
            input_image = input_image.rotate(angle, resample=Image.BICUBIC)
            target_image = target_image.rotate(angle, resample=Image.BICUBIC)

        if random.random() < 0.33:
            scale = 0.5 + random.random() * 1.5
            new_w = int(round(input_image.width * scale))
            new_h = int(round(input_image.height * scale))
            if new_w >= self.crop_size and new_h >= self.crop_size:
                input_image = input_image.resize((new_w, new_h), resample=Image.BICUBIC)
                target_image = target_image.resize((new_w, new_h), resample=Image.BICUBIC)

        if random.random() < 0.5:
            input_image = ImageOps.mirror(input_image)
            target_image = ImageOps.mirror(target_image)
        if random.random() < 0.5:
            input_image = ImageOps.flip(input_image)
            target_image = ImageOps.flip(target_image)

        if random.random() < 0.5:
            k = random.randint(1, 3)
            input_image = input_image.rotate(90 * k, resample=Image.NEAREST)
            target_image = target_image.rotate(90 * k, resample=Image.NEAREST)

        if random.random() < 0.1:
            arr_in = np.array(input_image)
            arr_tg = np.array(target_image)
            gray_in = np.mean(arr_in, axis=2, keepdims=True).astype(arr_in.dtype)
            gray_tg = np.mean(arr_tg, axis=2, keepdims=True).astype(arr_tg.dtype)
            arr_in = np.repeat(gray_in, 3, axis=2)
            arr_tg = np.repeat(gray_tg, 3, axis=2)
            input_image = Image.fromarray(arr_in)
            target_image = Image.fromarray(arr_tg)

        if random.random() < 0.7:
            arr_in = np.array(input_image).astype(np.float32) / 255.0
            arr_tg = np.array(target_image).astype(np.float32) / 255.0
            channel = random.randint(0, 2)
            minimum = min(arr_in.min(), arr_tg.min())
            offset = random.random() * 0.25 - random.random() * minimum
            arr_in[:, :, channel] = np.clip(arr_in[:, :, channel] + offset * (1.0 - arr_in[:, :, channel]), 0.0, 1.0)
            arr_tg[:, :, channel] = np.clip(arr_tg[:, :, channel] + offset * (1.0 - arr_tg[:, :, channel]), 0.0, 1.0)
            input_image = Image.fromarray((arr_in * 255.0).astype(np.uint8))
            target_image = Image.fromarray((arr_tg * 255.0).astype(np.uint8))

        if random.random() < 0.7:
            order = list(range(3))
            random.shuffle(order)
            arr_in = np.array(input_image)[:, :, order]
            arr_tg = np.array(target_image)[:, :, order]
            input_image = Image.fromarray(arr_in)
            target_image = Image.fromarray(arr_tg)

        return input_image, target_image

    def __getitem__(self, index: int) -> dict:
        input_image, target_image = self._open_pair(index)
        if self.augment:
            input_image, target_image = self._apply_starnet_augmentation(input_image, target_image)

        input_image, target_image = self._apply_crop(input_image, target_image)

        input_tensor = F.to_tensor(input_image)
        target_tensor = F.to_tensor(target_image)

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
