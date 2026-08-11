"""Rotary position embeddings (Su et al., 2021), Llama-style.

Instead of adding a learned/sinusoidal position vector to the token
embedding once (GPT-2 style), RoPE rotates each attention head's query/key
vectors by an angle proportional to sequence position. Two useful
consequences: q_m . k_n after rotation depends only on the relative offset
(m - n), and there's no fixed max-position embedding table to run out of --
the rotation is defined for any position.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim / 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x: (batch, n_head, seq_len, head_dim); cos/sin: (seq_len, head_dim / 2)."""
    cos = torch.cat([cos, cos], dim=-1)[None, None, :, :]
    sin = torch.cat([sin, sin], dim=-1)[None, None, :, :]
    return x * cos + _rotate_half(x) * sin
