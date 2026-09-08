"""Per-token router trace for one sequence -- the attribution signal that's
unique to this project: which expert each token activated, and with what
gate weight, at every layer. Reuses MoEGPT's existing
return_router_outputs=True path (no changes needed there) and
routing_analysis.per_token_routing_table's table format, so this and the
step-6 routing analysis stay directly comparable.

Dense models have no router, so get_router_trace returns None for them --
callers (the Explainer Agent, the dashboard) treat a None router_trace as
"this model has nothing to say here" rather than an error.
"""

from __future__ import annotations

import torch

from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.routing_analysis import per_token_routing_table


@torch.no_grad()
def get_router_trace(
    model: DenseGPT | MoEGPT, idx: torch.Tensor, token_strs: list[str]
) -> dict | None:
    """idx: (1, T) token ids for a single sequence. Returns the same dict
    shape as routing_analysis.per_token_routing_table (top1_experts, detail,
    tokens), or None if model has no router."""
    if not isinstance(model, MoEGPT):
        return None

    was_training = model.training
    model.eval()
    _, _, aux = model(idx, return_router_outputs=True)
    model.train(was_training)
    return per_token_routing_table(aux["router_outputs"], token_strs)


def dominant_expert_per_token(router_trace: dict, layer_idx: int = -1) -> list[int]:
    """Top-1 expert index per token at one layer (default: last layer,
    typically where routing specialization is sharpest) -- the series the
    Router-Attribution Agreement faithfulness metric (Phase 2) will
    rank-correlate against each explainer's attribution."""
    return router_trace["top1_experts"][layer_idx].tolist()
