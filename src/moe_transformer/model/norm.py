"""RMSNorm (Zhang & Sennrich, 2019) -- the Llama/Mixtral normalization layer.

Cheaper than LayerNorm: no mean-centering, just rescale by the RMS of the
activations. Computed in float32 regardless of input dtype since the
mean-of-squares reduction is where mixed-precision training loses precision.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(in_dtype) * self.weight
