"""Day 2 — V-B backbone: ResNet-18 variant with asymmetric strides over
the (time, velocity) axes. Ref. §5.2.

Design goal (§5.2): downsample TIME early and aggressively, preserve the
VELOCITY axis (the feature that separates walking/running) — 50 of the
100 velocity bins survive to the final map. Half width (channels
32/64/128/256 for d_enc=256) → ~4.7 GFLOPs/window (~1.8x the stock
ResNet-18 on this input), fitting the §8.4 budget on a T4.

Escalation (b), to be activated ONLY if the day-2 throughput gate fails
(§10.1): stride (2,2) at layer3 → final map ~11x25, ~2.9 GFLOPs
(~0.63x V-B). Every escalation goes in the split file changelog.
"""
from __future__ import annotations

import torch
import torch.nn as nn

Stride = tuple[int, int]


def _conv3x3(in_ch: int, out_ch: int, stride: Stride = (1, 1)) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet-18 basic block (two 3x3 convs + identity shortcut) with
    tuple strides, so time and velocity can be downsampled independently
    (§5.2). The 1x1 projection shortcut kicks in whenever the stride or
    the channel count changes."""

    def __init__(self, in_ch: int, out_ch: int, stride: Stride = (1, 1)) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != (1, 1) or in_ch != out_ch:
            self.downsample: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetVB(nn.Module):
    """V-B backbone (§5.2). Input (B, 1, time=340, velocity=100), output
    (B, d_enc) after global average pooling.

    Geometry for the reference input:
      stem  conv3x3 s(2,1) + maxpool3x3 s(2,1) -> 85 x 100 (time/4)
      layer1 s(1,1)  -> 85 x 100   (d_enc//8 channels)
      layer2 s(2,2)  -> 43 x 50    (d_enc//4)
      layer3 s(2,1)  -> 22 x 50    (d_enc//2)   [escalation_b: s(2,2) -> 22 x 25]
      layer4 s(2,1)  -> 11 x 50    (d_enc)      [escalation_b: -> 11 x 25]

    Channel widths derive from d_enc (d_enc//8 ... d_enc), no hardcoded
    counts. Heads (classifier / projection / adversary) live in
    heads.py, NOT here — the backbone is shared by C1-C4 (§5.3).
    """

    def __init__(self, d_enc: int = 256, in_channels: int = 1, escalation_b: bool = False) -> None:
        super().__init__()
        assert d_enc % 8 == 0, f"d_enc must be divisible by 8, got {d_enc}"
        self.d_enc = d_enc
        self.feature_dim = d_enc  # what the heads consume; sharp_like differs (fixed by geometry)
        self.escalation_b = escalation_b
        widths = (d_enc // 8, d_enc // 4, d_enc // 2, d_enc)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
        )
        self.layer1 = self._make_layer(widths[0], widths[0], stride=(1, 1))
        self.layer2 = self._make_layer(widths[0], widths[1], stride=(2, 2))
        self.layer3 = self._make_layer(widths[1], widths[2], stride=(2, 2) if escalation_b else (2, 1))
        self.layer4 = self._make_layer(widths[2], widths[3], stride=(2, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, stride: Stride) -> nn.Sequential:
        return nn.Sequential(BasicBlock(in_ch, out_ch, stride), BasicBlock(out_ch, out_ch))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-pooling feature map (B, d_enc, ~11, ~50) — exposed for
        shape checks and future spatial diagnostics."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(self.forward_features(x)).flatten(1)
