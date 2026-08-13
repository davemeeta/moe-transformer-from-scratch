"""The MoE layer: gating router, top-k dispatch, capacity limits, aux losses.

Each token is scored against every expert by a linear router, then sent to
its top-k highest-scoring experts. Unlike a dense FFN, compute here is
*bounded*: each expert has a fixed capacity (capacity_factor * expected
load), and once full it stops accepting tokens for the rest of the batch --
excess tokens are dropped for that expert, exactly the tradeoff Switch
Transformer and GShard make in production rather than letting one popular
expert blow up the compute budget. Overflow is resolved in token position
order (first-come-first-served): a token's earlier-in-sequence peers get
priority for a given expert's remaining slots. If only some of a token's
top-k picks are full, it still gets a (non-renormalized) contribution from
whichever picks weren't -- a token can be partially routed, not just
all-or-nothing.

Two auxiliary losses shape the router during training:
  - load-balancing loss (Switch/GShard): num_experts * sum_i(f_i * P_i),
    where f_i is the *hard* fraction of (token, slot) assignments expert i
    received (no gradient) and P_i is the *soft*, differentiable mean
    router probability mass on expert i. f_i tells the loss which experts
    are overused; P_i is what the gradient actually pushes on. This is how
    gradient reaches the router without differentiating through the
    discrete top-k choice itself.
  - router z-loss (ST-MoE): mean(logsumexp(router_logits)^2), keeps router
    logits from growing unboundedly large, for training stability.

Dispatch here is real (capacity-bounded) gather/scatter over plain PyTorch
tensor ops -- functionally what Switch Transformer/GShard describe, but not
what makes them fast in production. Megablocks, Tutel, and DeepSpeed-MoE
replace this per-expert loop with fused GPU kernels; our version does less
compute (tokens beyond capacity are genuinely skipped) but can still be
slower in wall-clock time on a GPU because a Python-level loop over experts
doesn't parallelize the way a fused kernel does. That gap is expected and
is exactly what those libraries exist to close.

Measured on this project's dev machine (Apple Silicon): the dynamic-shape
ops this dispatch relies on (torch.nonzero, advanced-indexing gather/
scatter -- shapes that depend on runtime routing decisions, not just
tensor rank) are dramatically worse on MPS than on CPU, and MPS timing
was observed to *degrade* step over step (~4s -> ~15s) rather than settle,
instead of the reverse GPU-faster-than-CPU pattern you'd expect. CPU ran
the same model at ~2.4s/step, stable. Train MoE models on CPU on this kind
of setup; MPS is fine for DenseGPT (no dynamic shapes involved) but not for
this dispatch mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_transformer.config import ModelConfig
from moe_transformer.init import init_weights
from moe_transformer.model.feedforward import FeedForward


@dataclass
class MoEOutput:
    output: torch.Tensor
    load_balancing_loss: torch.Tensor
    z_loss: torch.Tensor
    num_dropped_tokens: int
    expert_assignment_counts: torch.Tensor  # (num_experts,), pre-capacity hard counts
    topk_idx: torch.Tensor  # (N, k), each token's selected expert indices
    topk_weights: torch.Tensor  # (N, k), combine weights for those experts


class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor

        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForward(config, hidden_dim=config.expert_ffn_hidden_dim)
            for _ in range(config.num_experts)
        )

        self.apply(init_weights)

    def route(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Router decisions for a flat (N, n_embd) batch of tokens.

        Returns (logits, full_probs, topk_idx, topk_weights):
          - logits: (N, E) raw router scores.
          - full_probs: (N, E) softmax over *all* experts, differentiable --
            this is what the load-balancing loss's gradient actually flows
            through.
          - topk_idx: (N, k) indices of each token's selected experts.
          - topk_weights: (N, k) softmax over just the selected logits, so
            it sums to 1 per token by construction (equivalent to taking
            full_probs's top-k entries and renormalizing them, just computed
            more directly).
        """
        logits = self.gate(x_flat)
        full_probs = F.softmax(logits, dim=-1)
        topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        return logits, full_probs, topk_idx, topk_weights

    def forward(self, x: torch.Tensor) -> MoEOutput:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        N = x_flat.size(0)

        logits, full_probs, topk_idx, topk_weights = self.route(x_flat)

        capacity = max(1, int(self.capacity_factor * N * self.top_k / self.num_experts))

        output = torch.zeros_like(x_flat)
        num_dropped = 0
        expert_assignment_counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device
        )

        for e, expert in enumerate(self.experts):
            assigned = topk_idx == e  # (N, k)
            token_assigned = assigned.any(dim=-1)  # (N,)
            expert_assignment_counts[e] = token_assigned.sum()

            positions = torch.nonzero(token_assigned, as_tuple=True)[0]  # ascending order
            if positions.numel() > capacity:
                num_dropped += positions.numel() - capacity
                positions = positions[:capacity]  # first-come-first-served

            if positions.numel() == 0:
                continue

            rank_idx = assigned[positions].float().argmax(dim=-1)
            weights = topk_weights[positions, rank_idx]

            expert_out = expert(x_flat[positions])
            output[positions] += expert_out * weights.unsqueeze(-1)

        output = output.reshape(B, T, C)

        f_i = expert_assignment_counts.float() / (N * self.top_k)
        P_i = full_probs.mean(dim=0)
        load_balancing_loss = self.num_experts * torch.sum(f_i * P_i)

        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()

        return MoEOutput(
            output=output,
            load_balancing_loss=load_balancing_loss,
            z_loss=z_loss,
            num_dropped_tokens=num_dropped,
            expert_assignment_counts=expert_assignment_counts,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
