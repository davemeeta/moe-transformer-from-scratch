"""Necessity red-team attack: holds an explainer's protected BPE token
positions fixed and greedily substitutes the remaining ("unimportant")
positions with nearest-neighbor tokens from the model's own embedding
table, searching for the fewest edits that flip P(target_token).

Deletion/insertion and comprehensiveness/sufficiency (Phase 2) only ever
perturb the tokens an explainer already flagged as important. This is the
check they can't give you: if editing only the tokens an explainer called
*unimportant* is enough to flip the prediction anyway, the explainer missed
where the model's real sensitivity lived. Fewer edits needed = less
faithful; failing to flip within the edit budget = more faithful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.redteam.embedding_neighbors import nearest_neighbor_ids
from moe_transformer.xai.utils import get_next_token_logits


@dataclass
class RedTeamResult:
    protected_positions: list[int]
    editable_positions: list[int]
    original_prob: float
    final_prob: float
    flipped: bool
    num_edits: int
    edit_fraction: float
    substitutions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "protected_positions": self.protected_positions,
            "editable_positions": self.editable_positions,
            "original_prob": self.original_prob,
            "final_prob": self.final_prob,
            "flipped": self.flipped,
            "num_edits": self.num_edits,
            "edit_fraction": self.edit_fraction,
            "substitutions": self.substitutions,
        }


def necessity_attack(
    model: DenseGPT | MoEGPT,
    tokenizer: Tokenizer,
    device: str,
    token_ids: list[int],
    target_token_id: int,
    protected: set[int],
    flip_threshold: float = 0.5,
    top_n_neighbors: int = 8,
    max_edits: int = 6,
) -> RedTeamResult:
    was_training = model.training
    model.eval()
    embedding_weight = model.token_emb.weight

    @torch.no_grad()
    def prob_of(ids: list[int]) -> float:
        idx = torch.tensor([ids], device=device)
        logits = get_next_token_logits(model, idx)[0]
        return torch.softmax(logits, dim=-1)[target_token_id].item()

    working_ids = list(token_ids)
    original_prob = prob_of(working_ids)
    editable = [i for i in range(len(token_ids)) if i not in protected]
    candidates = {
        pos: nearest_neighbor_ids(embedding_weight, token_ids[pos], top_n_neighbors)
        for pos in editable
    }

    remaining = set(editable)
    substitutions: list[dict] = []
    current_prob = original_prob
    flipped = False
    budget = min(max_edits, len(editable))

    for _ in range(budget):
        best: tuple[float, int, int] | None = None  # (new_prob, position, candidate_id)
        for pos in remaining:
            for cand_id in candidates[pos]:
                trial_ids = list(working_ids)
                trial_ids[pos] = cand_id
                p = prob_of(trial_ids)
                if best is None or p < best[0]:
                    best = (p, pos, cand_id)
        if best is None:
            break

        p, pos, cand_id = best
        substitutions.append(
            {
                "position": pos,
                "original_token": tokenizer.decode([working_ids[pos]]),
                "replacement_token": tokenizer.decode([cand_id]),
            }
        )
        working_ids[pos] = cand_id
        remaining.discard(pos)
        current_prob = p

        if original_prob > 0 and (original_prob - current_prob) / original_prob >= flip_threshold:
            flipped = True
            break

    model.train(was_training)
    return RedTeamResult(
        protected_positions=sorted(protected),
        editable_positions=editable,
        original_prob=original_prob,
        final_prob=current_prob,
        flipped=flipped,
        num_edits=len(substitutions),
        edit_fraction=len(substitutions) / len(editable) if editable else 0.0,
        substitutions=substitutions,
    )
