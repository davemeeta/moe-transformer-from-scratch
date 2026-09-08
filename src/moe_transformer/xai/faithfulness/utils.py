"""Shared plumbing for every faithfulness metric: rank tokens by attribution
magnitude, and re-run the model with a subset of tokens kept.

Two eval_fn backends, matching the two token granularities the Explainer
Agent produces (see shap_explainer.py's module docstring): attention
rollout and router trace share this project's own BPE tokens, so masking
can drop token ids directly and re-run the model -- no re-tokenization.
SHAP/LIME attribute a masker's own word segmentation, so masking there
means dropping words from the reconstructed text and re-encoding it, same
as xai.utils.make_target_prob_fn already does for the explainers themselves.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.utils import get_next_token_logits, make_target_prob_fn

EvalFn = Callable[[set[int]], float]


def sorted_by_importance(values: list[float]) -> list[int]:
    """Token indices ordered by descending |attribution| -- the order both
    deletion and insertion walk through, one token at a time."""
    return sorted(range(len(values)), key=lambda i: -abs(values[i]))


def auc(fractions: list[float], probs: list[float]) -> float:
    return float(np.trapezoid(probs, fractions))


def make_bpe_eval_fn(
    model: DenseGPT | MoEGPT,
    tokenizer: Tokenizer,
    device: str,
    token_ids: list[int],
    target_token_id: int,
) -> EvalFn:
    was_training = model.training
    model.eval()

    @torch.no_grad()
    def eval_fn(kept_positions: set[int]) -> float:
        kept = [tid for i, tid in enumerate(token_ids) if i in kept_positions]
        if not kept:
            kept = [tokenizer.eot_token]
        idx = torch.tensor([kept], device=device)
        logits = get_next_token_logits(model, idx)[0]
        prob = torch.softmax(logits, dim=-1)[target_token_id].item()
        model.train(was_training)
        return prob

    return eval_fn


def make_word_eval_fn(
    model: DenseGPT | MoEGPT,
    tokenizer: Tokenizer,
    device: str,
    words: list[str],
    target_token_id: int,
    block_size: int,
) -> EvalFn:
    predict_fn = make_target_prob_fn(model, tokenizer, device, target_token_id, block_size)

    def eval_fn(kept_positions: set[int]) -> float:
        text = "".join(words[i] for i in sorted(kept_positions))
        return float(predict_fn([text])[0])

    return eval_fn
