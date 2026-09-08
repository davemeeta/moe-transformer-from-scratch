"""Faithfulness Agent: scores every explainer the Explainer Agent ran, using
metrics that check whether an attribution matches what actually moves the
model -- not just whether it looks plausible.

Reads state written by ExplainerAgent (token_ids, attention_importance,
shap/lime words+values, router_trace) and adds state["faithfulness"]: one
{deletion_auc, insertion_auc, comprehensiveness, sufficiency, original_prob}
dict per explainer, plus router_attribution_agreement on attention when the
model has a router.
"""

from __future__ import annotations

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.agents.base import Agent, AgentState
from moe_transformer.xai.faithfulness.comp_suff import comprehensiveness, sufficiency
from moe_transformer.xai.faithfulness.deletion_insertion import (
    deletion_curve,
    insertion_curve,
)
from moe_transformer.xai.faithfulness.router_attribution_agreement import (
    router_attribution_agreement,
)
from moe_transformer.xai.faithfulness.utils import EvalFn, auc, make_bpe_eval_fn, make_word_eval_fn


class FaithfulnessAgent(Agent):
    name = "faithfulness"

    def __init__(
        self,
        model: DenseGPT | MoEGPT,
        tokenizer: Tokenizer,
        device: str,
        block_size: int,
        comprehensiveness_k: float = 0.2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.comprehensiveness_k = comprehensiveness_k

    def run(self, state: AgentState) -> AgentState:
        target_token_id = state["target_token_id"]

        attention_eval_fn = make_bpe_eval_fn(
            self.model, self.tokenizer, self.device, state["token_ids"], target_token_id
        )
        attention_scores = self._score(state["attention_importance"], attention_eval_fn)
        if state["router_trace"] is not None:
            attention_scores["router_attribution_agreement"] = router_attribution_agreement(
                state["attention_importance"], state["router_trace"]
            )

        shap_eval_fn = make_word_eval_fn(
            self.model, self.tokenizer, self.device, state["shap_words"], target_token_id, self.block_size
        )
        shap_scores = self._score(state["shap_values"], shap_eval_fn)

        lime_eval_fn = make_word_eval_fn(
            self.model, self.tokenizer, self.device, state["lime_words"], target_token_id, self.block_size
        )
        lime_scores = self._score(state["lime_values"], lime_eval_fn)

        state["faithfulness"] = {
            "attention": attention_scores,
            "shap": shap_scores,
            "lime": lime_scores,
        }
        return state

    def _score(self, values: list[float], eval_fn: EvalFn) -> dict:
        fractions, deletion_probs = deletion_curve(values, eval_fn)
        _, insertion_probs = insertion_curve(values, eval_fn)
        original_prob = deletion_probs[0]
        return {
            "deletion_auc": auc(fractions, deletion_probs),
            "insertion_auc": auc(fractions, insertion_probs),
            "comprehensiveness": comprehensiveness(
                values, self.comprehensiveness_k, original_prob, eval_fn
            ),
            "sufficiency": sufficiency(values, self.comprehensiveness_k, original_prob, eval_fn),
            "original_prob": original_prob,
        }
