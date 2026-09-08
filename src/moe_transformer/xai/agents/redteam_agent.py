"""Red-Team Agent: runs the necessity attack (xai.redteam.search) against
each explainer's protected token set, so all three land in one comparable
table -- fewer edits needed to flip the prediction means that explainer's
"important" tokens weren't actually where the model's sensitivity lived.

Reads state written by ExplainerAgent (token_ids, target_token_id,
attention_importance, shap/lime words+values) and adds
state["redteam"]: one RedTeamResult per explainer.
"""

from __future__ import annotations

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.agents.base import Agent, AgentState
from moe_transformer.xai.redteam.protected_positions import (
    protected_positions_bpe,
    protected_positions_from_words,
)
from moe_transformer.xai.redteam.search import RedTeamResult, necessity_attack


class RedTeamAgent(Agent):
    name = "redteam"

    def __init__(
        self,
        model: DenseGPT | MoEGPT,
        tokenizer: Tokenizer,
        device: str,
        protected_k: float = 0.2,
        top_n_neighbors: int = 8,
        max_edits: int = 6,
        flip_threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.protected_k = protected_k
        self.top_n_neighbors = top_n_neighbors
        self.max_edits = max_edits
        self.flip_threshold = flip_threshold

    def run(self, state: AgentState) -> AgentState:
        token_ids = state["token_ids"]
        target_token_id = state["target_token_id"]

        protected_sets = {
            "attention": protected_positions_bpe(state["attention_importance"], self.protected_k),
            "shap": protected_positions_from_words(
                state["shap_words"], state["shap_values"], token_ids, self.tokenizer, self.protected_k
            ),
            "lime": protected_positions_from_words(
                state["lime_words"], state["lime_values"], token_ids, self.tokenizer, self.protected_k
            ),
        }

        results: dict[str, RedTeamResult] = {}
        for name, protected in protected_sets.items():
            results[name] = necessity_attack(
                self.model,
                self.tokenizer,
                self.device,
                token_ids,
                target_token_id,
                protected,
                flip_threshold=self.flip_threshold,
                top_n_neighbors=self.top_n_neighbors,
                max_edits=self.max_edits,
            )

        state["redteam"] = results
        return state
