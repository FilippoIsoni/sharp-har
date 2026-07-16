"""Day 4 — C0 reproduction network, faithful to the SHARP paper.
Ref. §5.1 (architecture, "faithful, no modifications"), §6-C0.

Architecture (TMC §4.1 figure; repo `francescamen/SHARP`
`Python_code/CSI_network.py`): three parallel branches over the
1@(340x100) input —

  A: maxpool 2x2 stride 2                                  -> 1@170x50
  B: conv 5@(2x2) stride 2, ReLU                           -> 5@170x50
  C: conv 3@(1x1) s1 -> conv 6@(2x2) s1 -> conv 9@(4x4) s2 -> 9@170x50

concat (15 ch) -> conv 3@(1x1) ReLU -> flatten -> dropout 0.2 -> dense.
The dense classifier is ActivityHead outside this module (train.py's
backbone/head split); `feature_dim` (= 3 * ceil(t/2) * ceil(v/2) =
25500 for 340x100) is what the head consumes — d_enc does NOT apply to
this backbone.

The reference implementation is Keras: convolutions use Keras-style
"same" padding (asymmetric for even kernels, extra row/col at
bottom/right), reproduced here explicitly so branch shapes match for
the concat.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SameConv2d(nn.Module):
    """Conv2d with Keras-style 'same' padding: output = ceil(in/stride),
    pad_total = max((out-1)*stride + k - in, 0), split begin = total//2,
    end = the remainder (Keras puts the extra pad at bottom/right)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride)
        self._k, self._s = kernel, stride

    def _pad(self, size: int) -> tuple[int, int]:
        out = math.ceil(size / self._s)
        total = max((out - 1) * self._s + self._k - size, 0)
        return total // 2, total - total // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top, bottom = self._pad(x.shape[2])
        left, right = self._pad(x.shape[3])
        return self.conv(F.pad(x, (left, right, top, bottom)))


class SharpLike(nn.Module):
    """C0 backbone (§5.1). forward: (B, 1, time, velocity) ->
    (B, feature_dim) post-dropout flattened features; the dense
    classifier is ActivityHead(feature_dim, n_att) outside.

    `d_enc` is accepted for the build_backbone(cfg) signature but does
    not parametrize this network (faithful reproduction): the feature
    size is fixed by the geometry via `feature_dim`.
    """

    def __init__(
        self, d_enc: int | None = None, time_steps: int = 340, velocity_bins: int = 100
    ) -> None:
        super().__init__()
        self.branch_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch_conv = nn.Sequential(_SameConv2d(1, 5, kernel=2, stride=2), nn.ReLU(inplace=True))
        self.branch_deep = nn.Sequential(
            _SameConv2d(1, 3, kernel=1, stride=1), nn.ReLU(inplace=True),
            _SameConv2d(3, 6, kernel=2, stride=1), nn.ReLU(inplace=True),
            _SameConv2d(6, 9, kernel=4, stride=2), nn.ReLU(inplace=True),
        )
        self.merge = nn.Sequential(_SameConv2d(1 + 5 + 9, 3, kernel=1, stride=1), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.feature_dim = 3 * math.ceil(time_steps / 2) * math.ceil(velocity_bins / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = self.merge(torch.cat(
            [self.branch_pool(x), self.branch_conv(x), self.branch_deep(x)], dim=1
        ))
        return self.dropout(merged.flatten(1))
