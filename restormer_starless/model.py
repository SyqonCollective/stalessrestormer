"""Restormer architecture tailored for star removal."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.project_in = nn.Sequential(
            nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1, groups=hidden_dim * 2, bias=bias),
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x = torch.nn.functional.gelu(x1) * x2
        return self.project_out(x)


class MultiDConvHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qkv = self.dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion: float,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MultiDConvHeadAttention(dim, num_heads=num_heads, bias=bias)
        self.drop_path1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, expansion=ffn_expansion, bias=bias)
        self.drop_path2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path1(x)
        x = x + res

        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_path2(x)
        return x + res


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x)


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            LayerNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion: float = 2.66,
        bias: bool = True,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        dim = int(dim * alpha)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        dims: List[int] = [dim, dim * 2, dim * 4, dim * 8]

        # encoder
        self.encoder_levels = nn.ModuleList()
        in_dim = dim
        for level, blocks in enumerate(num_blocks):
            stage = nn.Sequential(
                *[TransformerBlock(in_dim, heads[level], ffn_expansion, bias=bias) for _ in range(blocks)]
            )
            self.encoder_levels.append(stage)
            if level < len(num_blocks) - 1:
                self.encoder_levels.append(Downsample(in_dim, in_dim * 2))
                in_dim *= 2

        # bottleneck
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(dims[-1], heads[-1], ffn_expansion, bias=bias) for _ in range(num_blocks[-1])]
        )

        # decoder
        self.decoder_levels = nn.ModuleList()
        in_dim = dims[-1]
        for level in range(len(num_blocks) - 2, -1, -1):
            up = Upsample(in_dim, dims[level])
            reduce = nn.Conv2d(dims[level] * 2, dims[level], kernel_size=1, bias=bias)
            blocks = nn.Sequential(
                *[
                    TransformerBlock(
                        dims[level],
                        heads[level],
                        ffn_expansion,
                        bias=bias,
                    )
                    for _ in range(num_blocks[level])
                ]
            )
            self.decoder_levels.append(
                nn.ModuleDict({"up": up, "reduce": reduce, "blocks": blocks})
            )
            in_dim = dims[level]

        self.refinement = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion, bias=bias) for _ in range(num_refinement_blocks)]
        )
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        skips: List[torch.Tensor] = []
        idx = 0
        while idx < len(self.encoder_levels):
            stage = self.encoder_levels[idx]
            x = stage(x)
            if idx < len(self.encoder_levels) - 1:
                skips.append(x)
                idx += 1
                down = self.encoder_levels[idx]
                x = down(x)
            idx += 1

        x = self.bottleneck(x)

        for level, stage in enumerate(self.decoder_levels):
            x = stage["up"](x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = stage["reduce"](x)
            x = stage["blocks"](x)

        x = self.refinement(x)
        out = self.output(x)
        return torch.tanh(out)
