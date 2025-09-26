"""Loss functions for Restormer starless training."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import warnings


class PerceptualLoss(nn.Module):
    def __init__(self, layers: Iterable[str] | None = None) -> None:
        super().__init__()
        if layers is None:
            layers = ["relu1_1", "relu2_1", "relu3_1"]
        self.layers = list(layers)
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except Exception as exc:  # pragma: no cover - offline fallback
            warnings.warn(f"Falling back to randomly initialised VGG19 features ({exc})")
            vgg = models.vgg19(weights=None).features
        mapping = {
            "relu1_1": 1,
            "relu2_1": 6,
            "relu3_1": 11,
            "relu4_1": 20,
            "relu5_1": 29,
        }
        max_idx = max(mapping[layer] for layer in self.layers)
        self.vgg = vgg[: max_idx + 1]
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2
        mean = self.mean.to(pred_norm.device)
        std = self.std.to(pred_norm.device)
        pred_norm = (pred_norm - mean) / std
        target_norm = (target_norm - mean) / std

        loss = 0.0
        x = pred_norm
        y = target_norm
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            name = None
            if idx == 1:
                name = "relu1_1"
            elif idx == 6:
                name = "relu2_1"
            elif idx == 11:
                name = "relu3_1"
            elif idx == 20:
                name = "relu4_1"
            elif idx == 29:
                name = "relu5_1"
            if name and name in self.layers:
                loss = loss + F.l1_loss(x, y)
        return loss
