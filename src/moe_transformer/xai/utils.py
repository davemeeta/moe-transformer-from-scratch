"""Model-agnostic helpers shared by every explainer.

DenseGPT.forward returns (logits, loss); MoEGPT.forward returns
(logits, loss, aux). Every explainer needs next-token logits for arbitrary
perturbed text without caring which model kind it got, so that adapter lives
here once instead of duplicated in each explainer.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT


def get_next_token_logits(model: DenseGPT | MoEGPT, idx: torch.Tensor) -> torch.Tensor:
    """Runs a forward pass and returns logits for the next token: (B, vocab_size)."""
    out = model(idx)
    logits = out[0]
    return logits[:, -1, :]


@torch.no_grad()
def predict_top_token(
    model: DenseGPT | MoEGPT, tokenizer: Tokenizer, device: str, token_ids: list[int]
) -> tuple[int, str]:
    """The model's own top-1 next-token prediction for a token sequence --
    used as the default explanation target when the caller doesn't pin one."""
    idx = torch.tensor([token_ids], device=device)
    logits = get_next_token_logits(model, idx)
    top_id = int(logits[0].argmax().item())
    return top_id, tokenizer.decode([top_id])


def make_target_prob_fn(
    model: DenseGPT | MoEGPT,
    tokenizer: Tokenizer,
    device: str,
    target_token_id: int,
    block_size: int,
) -> Callable[[list[str]], np.ndarray]:
    """Builds the black-box function SHAP/LIME perturb around: raw text in,
    P(target_token | text) out. Both libraries mask/remove words from the
    original text and need to see how that probability moves in response --
    neither ever looks inside the model itself.

    Re-tokenizes each perturbed string from scratch rather than trying to
    keep it aligned with the original BPE token boundaries: SHAP/LIME do
    their own word-level masking on raw text, so the honest thing is to feed
    that text back through the same tokenizer path a real caller would use,
    including the case where masking empties the string entirely.
    """
    was_training = model.training
    model.eval()

    @torch.no_grad()
    def predict(texts: list[str]) -> np.ndarray:
        probs = np.zeros(len(texts), dtype=np.float64)
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text) or [tokenizer.eot_token]
            ids = ids[-block_size:]
            idx = torch.tensor([ids], device=device)
            logits = get_next_token_logits(model, idx)[0]
            probs[i] = torch.softmax(logits, dim=-1)[target_token_id].item()
        model.train(was_training)
        return probs

    return predict


def tokens_to_strs(tokenizer: Tokenizer, token_ids: list[int]) -> list[str]:
    """Decodes each token id individually so attribution values can be
    reported per-token (matches routing_analysis.per_token_routing_table's
    convention of one display string per token id)."""
    return [tokenizer.decode([tid]) for tid in token_ids]
