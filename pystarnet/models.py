"""Model definitions for PyStarNet."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        inner = channels // heads
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).reshape(b, self.heads, c // self.heads, h * w)
        k = self.key(x).reshape(b, self.heads, c // self.heads, h * w)
        v = self.value(x).reshape(b, self.heads, c // self.heads, h * w)
        attn = torch.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, c, h, w)
        return self.proj(out) + x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class StarGenerator(nn.Module):
    def __init__(self, base_channels: int = 64, num_res_blocks: int = 2, attention: bool = True) -> None:
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, padding=3),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )

        depth = 4
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        encoder_channels: List[int] = []
        channels = base_channels
        for stage in range(depth):
            blocks = [ResidualBlock(channels) for _ in range(num_res_blocks)]
            self.encoders.append(nn.Sequential(*blocks))
            encoder_channels.append(channels)
            if stage < depth - 1:
                self.downsamples.append(Downsample(channels, channels * 2))
                channels *= 2
            else:
                self.downsamples.append(nn.Identity())

        self.bottleneck = nn.Sequential(
            ResidualBlock(channels),
            SelfAttention2d(channels) if attention else nn.Identity(),
            ResidualBlock(channels),
        )

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        skip_channels = list(reversed(encoder_channels[:-1]))
        for skip in skip_channels:
            self.upsamples.append(Upsample(channels, skip))
            decoder_layers: List[nn.Module] = [
                nn.Conv2d(skip * 2, skip, 3, padding=1),
                nn.GroupNorm(8, skip),
                nn.SiLU(inplace=True),
            ]
            decoder_layers.extend(ResidualBlock(skip) for _ in range(num_res_blocks))
            self.decoders.append(nn.Sequential(*decoder_layers))
            channels = skip

        self.exit = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.entry(x)
        skips: List[torch.Tensor] = []
        for idx, (encoder, downsample) in enumerate(zip(self.encoders, self.downsamples)):
            h = encoder(h)
            if idx < len(self.encoders) - 1:
                skips.append(h)
            h = downsample(h)
        h = self.bottleneck(h)
        for idx, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            h = upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)
        starless = self.exit(h)
        # predict the full starless image instead of a residual to avoid reintroducing stars from x
        return torch.tanh(starless)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = 6
        channels = base_channels
        for idx in range(4):
            stride = 1 if idx == 3 else 2
            conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, channels, 4, stride=stride, padding=1)
            )
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = channels
            channels = min(channels * 2, base_channels * 8)
        self.body = nn.Sequential(*layers)
        self.head = nn.utils.spectral_norm(nn.Conv2d(in_channels, 1, 4, padding=1))

    def forward(self, input_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_image, target_image], dim=1)
        x = self.body(x)
        return self.head(x)


def build_models(base_channels: int = 64, num_res_blocks: int = 2, attention: bool = True) -> Tuple[nn.Module, nn.Module]:
    generator = StarGenerator(base_channels=base_channels, num_res_blocks=num_res_blocks, attention=attention)
    discriminator = PatchDiscriminator(base_channels=base_channels)
    return generator, discriminator
