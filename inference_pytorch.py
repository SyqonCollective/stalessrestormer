#!/usr/bin/env python
"""Inference utilities for PyStarNet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as F

from pystarnet.models import build_models


def load_generator(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    generator, _ = build_models()
    state = torch.load(checkpoint, map_location=device)
    if "generator" in state:
        generator.load_state_dict(state["generator"])
    else:
        generator.load_state_dict(state)
    generator.to(device)
    generator.eval()
    return generator


def prepare_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image)
    return tensor.unsqueeze(0) * 2 - 1


def save_image(tensor: torch.Tensor, path: Path) -> None:
    image = (tensor.clamp(-1, 1) + 1) / 2
    image = F.to_pil_image(image.squeeze(0))
    image.save(path)


def sliding_window_inference(
    generator: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    _, _, h, w = image.shape
    pad_h = (tile_size - h % stride) % stride
    pad_w = (tile_size - w % stride) % stride
    image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_h, padded_w = image.shape

    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)

    for top in range(0, padded_h - tile_size + 1, stride):
        for left in range(0, padded_w - tile_size + 1, stride):
            patch = image[:, :, top : top + tile_size, left : left + tile_size].to(device)
            with torch.no_grad():
                pred = generator(patch)
            output[:, :, top : top + tile_size, left : left + tile_size] += pred.detach().cpu()
            weight[:, :, top : top + tile_size, left : left + tile_size] += 1

    output = output / weight.clamp_min(1)
    return output[:, :, :h, :w]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyStarNet inference on an image")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--input", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Where to save the starless result")
    parser.add_argument("--tile-size", type=int, default=256, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = load_generator(args.checkpoint, device)
    image = prepare_image(args.input)
    output = sliding_window_inference(generator, image, args.tile_size, args.stride, device)
    save_image(output, args.output)


if __name__ == "__main__":
    main()
