"""Utility helpers for PyStarNet."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scaler_g: Optional[torch.cuda.amp.GradScaler],
    scaler_d: Optional[torch.cuda.amp.GradScaler],
    generator_ema: Optional[torch.nn.Module] = None,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "step": global_step,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "scaler_g": scaler_g.state_dict() if scaler_g else None,
            "scaler_d": scaler_d.state_dict() if scaler_d else None,
            "generator_ema": generator_ema.state_dict() if generator_ema else None,
        },
        path,
    )


def save_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a") as f:
        f.write(json.dumps(metrics) + "\n")


def save_preview(path: Path, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor) -> None:
    ensure_dir(path.parent)
    grid = torch.cat([inputs, outputs, targets], dim=0)
    grid = (grid + 1) / 2
    save_image(make_grid(grid, nrow=inputs.size(0)), path)


class MetricAverager:
    def __init__(self) -> None:
        self.reset()

    def update(self, metrics: Dict[str, torch.Tensor]) -> None:
        for key, value in metrics.items():
            val = value.detach().item() if torch.is_tensor(value) else float(value)
            total, count = self.storage.get(key, (0.0, 0))
            self.storage[key] = (total + val, count + 1)

    def compute(self) -> Dict[str, float]:
        return {key: total / count for key, (total, count) in self.storage.items() if count > 0}

    def reset(self) -> None:
        self.storage: Dict[str, Any] = {}
