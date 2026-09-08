"""Router-Attribution Agreement -- this project's own faithfulness metric,
not from the XAI literature: it only makes sense for a model that routes.

Deletion/insertion and comprehensiveness/sufficiency all ask "does this
attribution match the model's *output* behavior." This asks a different
question specific to MoE: does the attribution match the model's *internal*
routing behavior? If a token gets a high attention-attribution score, does
the router also treat that token as needing strong expert specialization
(a high top-1 gate weight), or are the two signals telling unrelated
stories? Rank-correlation (Kendall's tau) rather than raw value comparison,
since attention importance and gate weight live on unrelated scales -- only
their *ordering* is comparable.

Only defined for attributions aligned to this project's own BPE tokens
(attention rollout) -- SHAP/LIME attribute a masker's own word
segmentation, a different tokenization than the router ever saw, so their
per-token attributions can't be lined up against per-token gate weights
without inventing an alignment that doesn't actually exist.
"""

from __future__ import annotations

from scipy.stats import kendalltau


def router_gate_weight_per_token(router_trace: dict, layer_idx: int = -1) -> list[float]:
    """Top-1 gate weight per token at one layer -- how strongly the router
    committed to its top expert choice for that token."""
    layer_detail = router_trace["detail"][layer_idx]
    return [max(weight for _, weight in token_detail) for token_detail in layer_detail]


def router_attribution_agreement(
    attribution_values: list[float], router_trace: dict, layer_idx: int = -1
) -> float:
    gate_weights = router_gate_weight_per_token(router_trace, layer_idx)
    if len(attribution_values) != len(gate_weights):
        raise ValueError(
            f"attribution length {len(attribution_values)} != router trace length "
            f"{len(gate_weights)} -- Router-Attribution Agreement requires an "
            "attribution aligned to this project's own BPE tokens (attention "
            "rollout), not SHAP/LIME's word-level segmentation."
        )
    tau, _ = kendalltau([abs(v) for v in attribution_values], gate_weights)
    return 0.0 if tau != tau else float(tau)  # nan-safe: constant input -> nan
