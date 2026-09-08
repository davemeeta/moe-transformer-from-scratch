"""LIME attribution for one next-token prediction.

Same black-box framing as shap_explainer: LimeTextExplainer perturbs `text`
by dropping words and fits a local linear surrogate to how P(target_token)
responds. LIME's API is classification-shaped (predict_proba over class
labels), so the target probability is wrapped as a 2-class
[1 - p, p] distribution and we read off the "target" class's word weights.

explanation.as_list() only returns LIME's top-`num_features` words, reordered
by |weight| -- fine for display, useless for Phase 2's faithfulness masking,
which needs every token in its original left-to-right position (same
requirement shap_explainer's masker-segmentation output already satisfies).
So num_features is set to cover every word in the instance, and weights are
read back out through LIME's own positional segmentation
(domain_mapper.indexed_string.as_list) rather than as_list()'s reordered
view -- non-word tokens (punctuation, whitespace) get weight 0.0, since LIME
never perturbs them.
"""

from __future__ import annotations

import numpy as np
from lime.lime_text import IndexedString, LimeTextExplainer

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.utils import make_target_prob_fn

TARGET_CLASS_INDEX = 1


def explain_lime(
    model: DenseGPT | MoEGPT,
    tokenizer: Tokenizer,
    device: str,
    text: str,
    target_token_id: int,
    block_size: int,
    num_samples: int = 500,
) -> tuple[list[str], list[float]]:
    """Returns (tokens, lime_weights) -- tokens is LIME's full positional
    segmentation of `text` (words and separators, in original order),
    lime_weights[i] is word i's linear weight toward predicting the "target
    token" class (0.0 for separators)."""
    target_prob_fn = make_target_prob_fn(model, tokenizer, device, target_token_id, block_size)

    def predict_proba(texts: list[str]) -> np.ndarray:
        p = target_prob_fn(texts)
        return np.stack([1.0 - p, p], axis=1)

    num_features = max(1, len(IndexedString(text).inverse_vocab))
    explainer = LimeTextExplainer(class_names=["not_target", "target"])
    explanation = explainer.explain_instance(
        text,
        predict_proba,
        labels=[TARGET_CLASS_INDEX],
        num_features=num_features,
        num_samples=num_samples,
    )
    word_weight = dict(explanation.as_list(label=TARGET_CLASS_INDEX))
    tokens = list(explanation.domain_mapper.indexed_string.as_list)
    weights = [float(word_weight.get(tok, 0.0)) for tok in tokens]
    return tokens, weights
