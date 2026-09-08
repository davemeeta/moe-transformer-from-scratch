"""Attention-based attribution: capture per-layer attention weights without
touching model/block.py, then combine them into one importance-per-token
score via Attention Rollout (Abnar & Zuidema, 2020).

CausalSelfAttention.forward already supports return_attn_weights=True (see
model/attention.py), but TransformerBlock/MoEBlock call it internally without
ever passing that flag, and always as `x + self.attn(...)` -- a single
tensor. Rather than editing block.py to thread a new flag through both block
types just for explainability, we temporarily monkeypatch each attention
module's `forward` for the duration of one call: the patched version always
requests weights, stashes them, and still returns only the tensor the block
expects, so the model's forward pass runs completely unmodified.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from moe_transformer.models import DenseGPT, MoEGPT


@contextmanager
def capture_attention_weights(model: DenseGPT | MoEGPT):
    """Yields a list with one entry per layer, filled in during the next
    forward pass: attn_weights tensors of shape (B, n_head, T, T). Restores
    each attention module's original forward on exit (or on exception)."""
    attn_modules = [block.attn for block in model.blocks]
    captured: list[torch.Tensor | None] = [None] * len(attn_modules)
    originals = [m.forward for m in attn_modules]

    def make_patched(layer_idx: int, original):
        def patched(x, return_attn_weights: bool = False):
            out, weights = original(x, return_attn_weights=True)
            captured[layer_idx] = weights
            return out

        return patched

    for layer_idx, m in enumerate(attn_modules):
        m.forward = make_patched(layer_idx, originals[layer_idx])
    try:
        yield captured
    finally:
        for m, original in zip(attn_modules, originals):
            m.forward = original


def attention_rollout(attn_weights_per_layer: list[torch.Tensor]) -> torch.Tensor:
    """attn_weights_per_layer: one (1, n_head, T, T) tensor per layer, batch
    size 1. Returns a (T,) importance-over-input-tokens vector, summing to 1,
    for the *last* sequence position -- the one the next-token prediction is
    actually read from.

    Averages heads, then folds in the residual stream (x = x + attn(x)) by
    blending each layer's attention matrix with the identity before chaining
    matrix products across layers: without this, rollout would only capture
    what attention moves between tokens and ignore the "just pass this
    token's own representation forward" path every residual connection
    provides.
    """
    rollout = None
    for attn_weights in attn_weights_per_layer:
        layer_attn = attn_weights[0].mean(dim=0)  # (T, T), averaged over heads
        T = layer_attn.shape[0]
        identity = torch.eye(T, device=layer_attn.device, dtype=layer_attn.dtype)
        residual_attn = 0.5 * layer_attn + 0.5 * identity
        rollout = residual_attn if rollout is None else residual_attn @ rollout
    last_token_importance = rollout[-1]
    return last_token_importance / last_token_importance.sum()
