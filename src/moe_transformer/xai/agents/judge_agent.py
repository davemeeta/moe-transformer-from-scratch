"""Judge Agent: an independent read on whether each explanation *sounds*
plausible to a human, checked against the faithfulness/red-team evidence
earlier agents already computed. The two signals are deliberately kept
independent -- plausibility never looks at the faithfulness numbers, and
the numbers never look at the plausibility score -- so a plausible-sounding
explanation that fails faithfulness (or a faithful one that reads oddly) is
the headline finding this agent surfaces, not something either signal could
report alone.

Runs on a local Ollama model rather than a paid API, per this project's
zero-cost constraint (see README/project writeup) -- `ollama serve` with a
model pulled (e.g. `ollama pull llama3.2:3b`) is the only setup required.
"""

from __future__ import annotations

import json
from typing import Callable

import requests

from moe_transformer.xai.agents.base import Agent, AgentState

OLLAMA_URL = "http://localhost:11434/api/generate"

PlausibilityFn = Callable[[str, str, list[str], list[float]], dict]


def _top_k_display(words: list[str], values: list[float], k: int = 5) -> str:
    order = sorted(range(len(values)), key=lambda i: -abs(values[i]))[:k]
    return ", ".join(f"{words[i]!r} ({values[i]:+.3f})" for i in order)


def query_ollama(model: str, prompt: str, timeout: float = 60.0) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


class JudgeAgent(Agent):
    name = "judge"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        flip_edit_fraction_threshold: float = 0.2,
        plausibility_fn: PlausibilityFn | None = None,
    ):
        self.model = model
        self.flip_edit_fraction_threshold = flip_edit_fraction_threshold
        # Injectable for tests (avoids a real Ollama call); defaults to the
        # real local-LLM read for actual runs.
        self._plausibility_fn = plausibility_fn or self._rate_plausibility

    def run(self, state: AgentState) -> AgentState:
        prompt_text = state["text"]
        target = state["target_token_str"]
        attribution_by_explainer = {
            "attention": (state["tokens"], state["attention_importance"]),
            "shap": (state["shap_words"], state["shap_values"]),
            "lime": (state["lime_words"], state["lime_values"]),
        }

        # Plausibility never depends on the red-team budget, so a re-run
        # after a deeper Red-Team retry (see orchestrator.py) reuses
        # whatever was already rated here instead of re-querying the LLM.
        plausibility = state.get("judge", {}).get("plausibility", {})
        verdicts: dict[str, dict] = {}
        for name, (words, values) in attribution_by_explainer.items():
            if name not in plausibility:
                plausibility[name] = self._plausibility_fn(prompt_text, target, words, values)

            redteam_result = state["redteam"][name]
            faithfulness_confidence = self._faithfulness_confidence(redteam_result)
            score = plausibility[name]["score"]
            divergent = (score >= 4 and faithfulness_confidence == "unfaithful") or (
                score <= 2 and faithfulness_confidence == "faithful"
            )

            verdicts[name] = {
                "plausibility_score": score,
                "plausibility_reasoning": plausibility[name]["reasoning"],
                "faithfulness_confidence": faithfulness_confidence,
                "divergent": divergent,
                "needs_deeper_redteam": not redteam_result.flipped,
            }

        state["judge"] = {"plausibility": plausibility, "verdicts": verdicts}
        return state

    def _faithfulness_confidence(self, redteam_result) -> str:
        """A cheap deterministic read on the same red-team result the Judge
        cross-checks plausibility against: a prediction broken by editing
        only "unimportant" tokens, with very few edits, means the explainer
        missed where the model's sensitivity actually lives (unfaithful). A
        red-team pass that couldn't break it at all is the strongest
        positive signal available (faithful). Anything else is genuinely
        mixed rather than confidently one or the other."""
        if redteam_result.flipped and redteam_result.edit_fraction <= self.flip_edit_fraction_threshold:
            return "unfaithful"
        if not redteam_result.flipped:
            return "faithful"
        return "mixed"

    def _rate_plausibility(
        self, prompt_text: str, target: str, words: list[str], values: list[float]
    ) -> dict:
        top_k = _top_k_display(words, values, k=5)
        prompt = (
            "A language model was given this text and predicted the next "
            f"token {target!r}.\n\nText: {prompt_text!r}\n\n"
            "An explanation method claims these tokens (with signed "
            f"importance toward that prediction) were most responsible: {top_k}.\n\n"
            "Rate how plausible this explanation sounds to a human reader "
            "on a 1-5 scale (1 = makes no sense, 5 = obviously correct), and "
            "give one sentence of reasoning. Respond ONLY with JSON in the "
            'form {"score": <int 1-5>, "reasoning": "<one sentence>"}'
        )
        try:
            raw = query_ollama(self.model, prompt)
            parsed = json.loads(raw)
            return {"score": int(parsed["score"]), "reasoning": str(parsed["reasoning"])}
        except Exception as e:
            return {
                "score": 3,
                "reasoning": f"judge unavailable ({type(e).__name__}) -- defaulted to neutral",
            }
