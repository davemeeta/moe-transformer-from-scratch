"""Manual multi-head causal self-attention.

Every step of scaled dot-product attention (QK^T, scale, causal mask,
softmax, weighted sum with V) is written out explicitly rather than calling
F.scaled_dot_product_attention. This is slower than PyTorch's fused/flash
attention kernel (and much slower than a real fused CUDA kernel), but keeps
attention weights inspectable -- useful later when we visualize routing
alongside attention, and it's the whole point of building this by hand.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_transformer.config import ModelConfig
from moe_transformer.init import init_weights
from moe_transformer.model.rope import RotaryEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.rope = RotaryEmbedding(
            self.head_dim, config.block_size, theta=config.rope_theta
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.apply(init_weights)

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(
            ~self.causal_mask[:T, :T], float("-inf")
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        if return_attn_weights:
            return out, attn_weights
        return out
