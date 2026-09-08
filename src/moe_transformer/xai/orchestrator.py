"""RouteLens Orchestrator: the one piece of real multi-agent control flow
in this pipeline (see agents/base.py's module docstring for why this is a
plain state machine, not a framework).

Runs Explainer -> Faithfulness -> Red-Team (cheap budget) -> Judge -> Report
in sequence, then the pipeline's one conditional edge: any explainer whose
red-team attack didn't flip the prediction within the cheap budget is
inconclusive -- absence of evidence isn't evidence of absence, it might
simply not have searched hard enough -- so the Orchestrator re-runs *only
those* explainers' Red-Team attack with a larger edit/neighbor budget and
re-derives the Judge's verdict for them, without re-querying the LLM for
plausibility a second time (that read doesn't depend on the red-team
budget at all, see judge_agent.py).
"""

from __future__ import annotations

from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.agents.base import AgentState
from moe_transformer.xai.agents.explainer_agent import ExplainerAgent
from moe_transformer.xai.agents.faithfulness_agent import FaithfulnessAgent
from moe_transformer.xai.agents.judge_agent import JudgeAgent, PlausibilityFn
from moe_transformer.xai.agents.redteam_agent import RedTeamAgent
from moe_transformer.xai.agents.report_agent import ReportAgent


class Orchestrator:
    def __init__(
        self,
        model: DenseGPT | MoEGPT,
        tokenizer: Tokenizer,
        device: str,
        block_size: int,
        shap_max_evals: int | str = "auto",
        lime_num_samples: int = 500,
        comprehensiveness_k: float = 0.2,
        redteam_protected_k: float = 0.2,
        redteam_top_n_neighbors: int = 8,
        redteam_max_edits: int = 6,
        redteam_flip_threshold: float = 0.5,
        deep_max_edits: int = 12,
        deep_top_n_neighbors: int = 16,
        judge_model: str = "llama3.2:3b",
        judge_plausibility_fn: PlausibilityFn | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.redteam_protected_k = redteam_protected_k
        self.redteam_flip_threshold = redteam_flip_threshold
        self.deep_max_edits = deep_max_edits
        self.deep_top_n_neighbors = deep_top_n_neighbors

        self.explainer = ExplainerAgent(
            model, tokenizer, device, block_size,
            shap_max_evals=shap_max_evals, lime_num_samples=lime_num_samples,
        )
        self.faithfulness = FaithfulnessAgent(
            model, tokenizer, device, block_size, comprehensiveness_k=comprehensiveness_k
        )
        self.redteam = RedTeamAgent(
            model, tokenizer, device,
            protected_k=redteam_protected_k,
            top_n_neighbors=redteam_top_n_neighbors,
            max_edits=redteam_max_edits,
            flip_threshold=redteam_flip_threshold,
        )
        self.judge = JudgeAgent(model=judge_model, plausibility_fn=judge_plausibility_fn)
        self.report = ReportAgent()

    def run(self, text: str, target_token_id: int | None = None) -> AgentState:
        state: AgentState = {"text": text, "target_token_id": target_token_id}
        state = self.explainer.run(state)
        state = self.faithfulness.run(state)
        state = self.redteam.run(state)
        state = self.judge.run(state)

        retry_names = [
            name for name, verdict in state["judge"]["verdicts"].items()
            if verdict["needs_deeper_redteam"]
        ]
        state["redteam_deep_retry"] = retry_names
        if retry_names:
            deep_redteam = RedTeamAgent(
                self.model, self.tokenizer, self.device,
                protected_k=self.redteam_protected_k,
                top_n_neighbors=self.deep_top_n_neighbors,
                max_edits=self.deep_max_edits,
                flip_threshold=self.redteam_flip_threshold,
            )
            deep_state = deep_redteam.run(dict(state))
            for name in retry_names:
                state["redteam"][name] = deep_state["redteam"][name]
            state = self.judge.run(state)

        state = self.report.run(state)
        return state
