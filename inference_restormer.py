#!/usr/bin/env python
"""Sliding-window inference for the Restormer starless model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import torch
from torchvision.transforms import functional as F
from PIL import Image

from restormer_starless import Restormer


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = Restormer()
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def prepare_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = F.to_tensor(image)
    return tensor.unsqueeze(0) * 2 - 1


def save_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    image = F.to_pil_image(tensor.squeeze(0))
    image.save(path)


def sliding_window(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile: int,
    stride: int,
    device: torch.device,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> torch.Tensor:
    _, _, h, w = image.shape
    pad_h = (tile - h % stride) % stride
    pad_w = (tile - w % stride) % stride
    image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H, W = image.shape

    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)

    total = ((H - tile) // stride + 1) * ((W - tile) // stride + 1)
    done = 0

    for top in range(0, H - tile + 1, stride):
        for left in range(0, W - tile + 1, stride):
            patch = image[:, :, top : top + tile, left : left + tile].to(device)
            with torch.no_grad():
                pred = model(patch)
            pred = pred.detach().cpu()
            output[:, :, top : top + tile, left : left + tile] += pred
            weight[:, :, top : top + tile, left : left + tile] += 1
            done += 1
            if progress_cb:
                progress_cb(done / total)

    output = output / weight.clamp_min(1.0)
    return output[:, :, :h, :w]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Restormer starless inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Restormer] Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print(f"[Restormer] Preparing image {args.input}")
    image = prepare_image(args.input)

    def cb(value: float) -> None:
        if args.show_progress:
            print(f"\r[Restormer] Progress {value:.1%}", end="", flush=True)

    print(
        f"[Restormer] Running sliding window (tile={args.tile}, stride={args.stride})"
    )
    output = sliding_window(model, image, args.tile, args.stride, device, cb if args.show_progress else None)
    if args.show_progress:
        print("\r[Restormer] Progress 100.0%")
    save_image(output, args.output)
    print(f"[Restormer] Saved starless image to {args.output}")


if __name__ == "__main__":
    main()
