"""Which BPE token positions the necessity attack (search.py) must leave
untouched, per explainer.

Attention rollout already reports importance per BPE token, so its top-k
positions are used directly. SHAP/LIME report importance per word (their
own masker segmentation, a different tokenization) -- but unlike
Router-Attribution Agreement, which would have to compare *magnitudes*
across two genuinely incompatible scales (attention weight vs. router gate
weight) and is deliberately restricted to attention only for that reason,
protecting a token from attack is a purely positional question: is it
inside one of the top-k words this explainer flagged? That membership fact
survives the tokenization mismatch cleanly -- no magnitude comparison
involved -- so SHAP/LIME get BPE-level protected sets too, via character-
span overlap with their own top-k words.
"""

from __future__ import annotations

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.xai.faithfulness.utils import sorted_by_importance


def _char_spans(strings: list[str]) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    for s in strings:
        spans.append((pos, pos + len(s)))
        pos += len(s)
    return spans


def protected_positions_bpe(values: list[float], k_fraction: float) -> set[int]:
    """Top-k BPE token positions directly -- for attributions already
    reported per BPE token (attention rollout)."""
    n = len(values)
    k = max(1, round(k_fraction * n))
    return set(sorted_by_importance(values)[:k])


def protected_positions_from_words(
    words: list[str],
    values: list[float],
    token_ids: list[int],
    tokenizer: Tokenizer,
    k_fraction: float,
) -> set[int]:
    """BPE token positions covered by an explainer's top-k words (SHAP/LIME),
    by character-span overlap between the word segmentation and the BPE
    tokenization of the same underlying text."""
    n_words = len(words)
    k = max(1, round(k_fraction * n_words))
    top_word_idx = set(sorted_by_importance(values)[:k])

    word_spans = _char_spans(words)
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]
    token_spans = _char_spans(token_strs)

    reconstructed_words = "".join(words)
    reconstructed_tokens = "".join(token_strs)
    if not reconstructed_tokens.startswith(reconstructed_words):
        # Exact match covers LIME (its IndexedString reconstructs losslessly).
        # A strict prefix covers SHAP's default Text masker, which silently
        # drops trailing text when the input doesn't end in whitespace (e.g.
        # "you?" loses the "?") -- a quirk of its own tokenizer regex, not
        # something under our control. Either way, any BPE token beyond the
        # word segmentation's actual coverage just isn't eligible for
        # protection from that explainer, which is the conservative
        # (under- not over-protecting) side to fail on.
        raise ValueError(
            "word segmentation and BPE tokenization reconstruct incompatible "
            "text -- cannot align protected positions"
        )

    protected: set[int] = set()
    for t_idx, (t_start, t_end) in enumerate(token_spans):
        for w_idx in top_word_idx:
            w_start, w_end = word_spans[w_idx]
            if w_start < t_end and w_end > t_start:
                protected.add(t_idx)
                break
    return protected
