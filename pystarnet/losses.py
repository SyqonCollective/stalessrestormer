"""Loss functions for PyStarNet."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn
from torch.nn import functional as F
import warnings
from torchvision import models


class HingeGANLoss(nn.Module):
    def forward_discriminator(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        loss_real = torch.mean(F.relu(1.0 - real_logits))
        loss_fake = torch.mean(F.relu(1.0 + fake_logits))
        return loss_real + loss_fake

    def forward_generator(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -torch.mean(fake_logits)


class PerceptualLoss(nn.Module):
    def __init__(self, layers: Iterable[str]) -> None:
        super().__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except Exception as exc:  # pragma: no cover - fallback for offline environments
            warnings.warn(f"Falling back to randomly initialised VGG19 features ({exc})")
            vgg = models.vgg19(weights=None).features
        self.layers = list(layers)
        self.blocks = nn.ModuleList()
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
            param.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2
        device = pred.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        pred_norm = (pred_norm - mean) / std
        target_norm = (target_norm - mean) / std

        loss = 0.0
        x = pred_norm
        y = target_norm
        features = {}
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


def generator_loss(
    fake_logits: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    hinge: HingeGANLoss,
    perceptual: PerceptualLoss,
    l1_weight: float,
    gan_weight: float,
    perceptual_weight: float,
) -> Dict[str, torch.Tensor]:
    loss_gan = hinge.forward_generator(fake_logits) * gan_weight
    loss_l1 = F.l1_loss(prediction, target) * l1_weight
    loss_perc = perceptual(prediction, target) * perceptual_weight
    total = loss_gan + loss_l1 + loss_perc
    return {
        "total": total,
        "gan": loss_gan.detach(),
        "l1": loss_l1.detach(),
        "perceptual": loss_perc.detach(),
    }


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    hinge: HingeGANLoss,
) -> torch.Tensor:
    return hinge.forward_discriminator(real_logits, fake_logits)
