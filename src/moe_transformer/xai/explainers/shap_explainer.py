"""SHAP attribution for one next-token prediction.

Treats the model as a black box: shap.maskers.Text masks out words from the
input and Explainer's partition algorithm (Owen values over a hierarchical
word split) attributes how each masked-out word moved P(target_token). Note
this operates on the masker's own word-level segmentation, not this
project's BPE tokens -- attention/router attributions are reported per BPE
token, SHAP/LIME per word; Phase 4's report agent displays them separately
rather than forcing a false alignment between the two.
"""

from __future__ import annotations

import shap

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.utils import make_target_prob_fn


def explain_shap(
    model: DenseGPT | MoEGPT,
    tokenizer: Tokenizer,
    device: str,
    text: str,
    target_token_id: int,
    block_size: int,
    max_evals: int | str = "auto",
) -> tuple[list[str], list[float]]:
    """Returns (words, shap_values) -- words is the masker's own
    segmentation of `text`, shap_values[i] is word i's signed contribution to
    P(target_token_id | text)."""
    predict_fn = make_target_prob_fn(model, tokenizer, device, target_token_id, block_size)
    masker = shap.maskers.Text()
    explainer = shap.Explainer(predict_fn, masker, algorithm="partition")
    explanation = explainer([text], max_evals=max_evals, silent=True)[0]
    return list(explanation.data), [float(v) for v in explanation.values]
