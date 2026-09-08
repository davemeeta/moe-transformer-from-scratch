"""Substitution candidates for the necessity attack (search.py), drawn from
the model's own tied token embedding table rather than an external synonym
resource (WordNet, a paraphrase model, ...). Free, offline, and arguably
more honest for this purpose: it's the model's own notion of which tokens
are interchangeable, not a human's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nearest_neighbor_ids(embedding_weight: torch.Tensor, token_id: int, top_n: int) -> list[int]:
    """top_n token ids closest to token_id in embedding space (cosine
    similarity), excluding token_id itself."""
    with torch.no_grad():
        vec = embedding_weight[token_id].unsqueeze(0)
        sims = F.cosine_similarity(embedding_weight, vec, dim=-1)
        sims[token_id] = -float("inf")
        top = torch.topk(sims, min(top_n, sims.numel() - 1)).indices
    return top.tolist()
