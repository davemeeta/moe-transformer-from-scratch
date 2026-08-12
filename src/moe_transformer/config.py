"""Model hyperparameters shared by every transformer block (dense and MoE)."""

from __future__ import annotations

from dataclasses import dataclass


def swiglu_hidden_dim(n_embd: int, multiple_of: int = 256) -> int:
    """Llama-style SwiGLU hidden size: ~2/3 of the usual 4x, rounded up.

    The 2/3 factor keeps a SwiGLU FFN (3 weight matrices) roughly
    parameter-matched to a GELU FFN (2 weight matrices) at the same
    hidden_dim = 4 * n_embd.
    """
    hidden_dim = int(2 * (4 * n_embd) / 3)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 256
    dropout: float = 0.1
    bias: bool = False
    ffn_hidden_dim: int | None = None
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    # MoE-specific (unused by DenseGPT)
    num_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.top_k <= self.num_experts, "top_k cannot exceed num_experts"
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = swiglu_hidden_dim(self.n_embd)
