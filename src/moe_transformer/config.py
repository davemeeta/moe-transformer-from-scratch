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


def moe_expert_hidden_dim(dense_hidden_dim: int, top_k: int, multiple_of: int = 64) -> int:
    """Per-expert SwiGLU hidden dim, sized so a token's top_k active experts
    do roughly the same FFN compute as one dense forward at dense_hidden_dim
    -- the "iso-FLOP" comparison that makes a dense-vs-MoE comparison mean
    something (more *total* params, same *active* params per token). At
    top_k=1 this returns dense_hidden_dim exactly (an expert is the whole
    dense FFN, as in Switch Transformer).

    Rounds *down* to a multiple of `multiple_of`, so active compute never
    exceeds the dense baseline, only matches or slightly undershoots it.
    """
    return max(multiple_of, (dense_hidden_dim // top_k // multiple_of) * multiple_of)


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
    expert_ffn_hidden_dim: int | None = None
    aux_loss_weight: float = 0.01  # Switch Transformer default
    router_z_loss_weight: float = 0.001  # ST-MoE default

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.top_k <= self.num_experts, "top_k cannot exceed num_experts"
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = swiglu_hidden_dim(self.n_embd)
        if self.expert_ffn_hidden_dim is None:
            self.expert_ffn_hidden_dim = moe_expert_hidden_dim(self.ffn_hidden_dim, self.top_k)
