"""SwiGLU feed-forward block (Shazeer, 2020), Llama/Mixtral-style.

Gated variant of the classic 2-matrix GELU FFN: a "gate" branch (SiLU
activation) elementwise-multiplies an "up" branch before the "down"
projection back to n_embd. This is also the exact module we'll reuse
per-expert in the MoE layer (step 4) -- an expert is just one of these.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_transformer.config import ModelConfig
from moe_transformer.init import init_weights


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
