"""Explainer Agent: runs every attribution method on one next-token
prediction and writes their raw output into shared state for downstream
agents (Faithfulness, Red-Team, Judge, Report -- later phases) to consume.

Explains state["text"] -> P(state["target_token_id"]); if no target is
pinned, defaults to the model's own top-1 next-token prediction, so
`ExplainerAgent(...).run({"text": prompt})` works with no other setup.
"""

from __future__ import annotations

import torch

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.agents.base import Agent, AgentState
from moe_transformer.xai.explainers.attention import (
    attention_rollout,
    capture_attention_weights,
)
from moe_transformer.xai.explainers.lime_explainer import explain_lime
from moe_transformer.xai.explainers.router_trace import get_router_trace
from moe_transformer.xai.explainers.shap_explainer import explain_shap
from moe_transformer.xai.utils import predict_top_token, tokens_to_strs


class ExplainerAgent(Agent):
    name = "explainer"

    def __init__(
        self,
        model: DenseGPT | MoEGPT,
        tokenizer: Tokenizer,
        device: str,
        block_size: int,
        shap_max_evals: int | str = "auto",
        lime_num_samples: int = 500,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.shap_max_evals = shap_max_evals
        self.lime_num_samples = lime_num_samples

    def run(self, state: AgentState) -> AgentState:
        text = state["text"]
        token_ids = self.tokenizer.encode(text)[-self.block_size :] or [
            self.tokenizer.eot_token
        ]
        idx = torch.tensor([token_ids], device=self.device)

        target_token_id = state.get("target_token_id")
        if target_token_id is None:
            target_token_id, target_token_str = predict_top_token(
                self.model, self.tokenizer, self.device, token_ids
            )
        else:
            target_token_str = self.tokenizer.decode([target_token_id])

        attention_importance = self._run_attention(idx)
        router_trace = get_router_trace(
            self.model, idx, tokens_to_strs(self.tokenizer, token_ids)
        )
        shap_words, shap_values = explain_shap(
            self.model,
            self.tokenizer,
            self.device,
            text,
            target_token_id,
            self.block_size,
            self.shap_max_evals,
        )
        lime_words, lime_values = explain_lime(
            self.model,
            self.tokenizer,
            self.device,
            text,
            target_token_id,
            self.block_size,
            self.lime_num_samples,
        )

        state.update(
            {
                "token_ids": token_ids,
                "tokens": tokens_to_strs(self.tokenizer, token_ids),
                "target_token_id": target_token_id,
                "target_token_str": target_token_str,
                "attention_importance": attention_importance.tolist(),
                "router_trace": router_trace,
                "shap_words": shap_words,
                "shap_values": shap_values,
                "lime_words": lime_words,
                "lime_values": lime_values,
            }
        )
        return state

    def _run_attention(self, idx: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        with capture_attention_weights(self.model) as captured:
            with torch.no_grad():
                self.model(idx)
        self.model.train(was_training)
        return attention_rollout(captured)
