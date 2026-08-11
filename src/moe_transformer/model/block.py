"""Pre-norm transformer block: x + Attn(Norm(x)), then x + FFN(Norm(x))."""

from __future__ import annotations

import torch
import torch.nn as nn

from moe_transformer.config import ModelConfig
from moe_transformer.model.attention import CausalSelfAttention
from moe_transformer.model.feedforward import FeedForward
from moe_transformer.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
