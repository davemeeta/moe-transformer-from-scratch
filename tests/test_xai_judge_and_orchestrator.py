"""Tests for JudgeAgent and Orchestrator use an injected plausibility_fn
rather than a real Ollama call -- fast and hermetic, and doesn't require
Ollama to be installed/running just to run the test suite. The real local-
LLM path is exercised manually via the CLI (run_explainer.py), not here.
"""

import pytest

from moe_transformer.config import ModelConfig
from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import MoEGPT
from moe_transformer.xai.agents.explainer_agent import ExplainerAgent
from moe_transformer.xai.agents.faithfulness_agent import FaithfulnessAgent
from moe_transformer.xai.agents.judge_agent import JudgeAgent
from moe_transformer.xai.agents.redteam_agent import RedTeamAgent
from moe_transformer.xai.orchestrator import Orchestrator
from moe_transformer.xai.redteam.search import RedTeamResult

TOKENIZER = Tokenizer()


def make_config(**overrides):
    defaults = dict(
        vocab_size=TOKENIZER.vocab_size,
        n_embd=16,
        n_head=2,
        n_layer=2,
        block_size=16,
        dropout=0.0,
        num_experts=4,
        top_k=2,
        capacity_factor=100.0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def fixed_plausibility_fn(score: int, reasoning: str = "stub reasoning"):
    def fn(prompt_text, target, words, values):
        return {"score": score, "reasoning": reasoning}
    return fn


def make_redteam_result(flipped: bool, edit_fraction: float) -> RedTeamResult:
    return RedTeamResult(
        protected_positions=[0],
        editable_positions=[1, 2],
        original_prob=0.5,
        final_prob=0.1 if flipped else 0.5,
        flipped=flipped,
        num_edits=1 if flipped else 0,
        edit_fraction=edit_fraction,
        substitutions=[],
    )


def base_state():
    return {
        "text": "Hello world",
        "target_token_str": "!",
        "tokens": ["Hello", " world"],
        "attention_importance": [0.9, 0.1],
        "shap_words": ["Hello", " world"],
        "shap_values": [0.8, 0.2],
        "lime_words": ["Hello", " world"],
        "lime_values": [0.7, 0.3],
    }


def test_judge_flags_divergence_when_plausible_but_unfaithful():
    state = base_state()
    state["redteam"] = {
        "attention": make_redteam_result(flipped=True, edit_fraction=0.05),
        "shap": make_redteam_result(flipped=True, edit_fraction=0.05),
        "lime": make_redteam_result(flipped=True, edit_fraction=0.05),
    }
    judge = JudgeAgent(plausibility_fn=fixed_plausibility_fn(score=5))
    state = judge.run(state)

    for verdict in state["judge"]["verdicts"].values():
        assert verdict["faithfulness_confidence"] == "unfaithful"
        assert verdict["divergent"] is True


def test_judge_no_divergence_when_plausible_and_faithful():
    state = base_state()
    state["redteam"] = {
        "attention": make_redteam_result(flipped=False, edit_fraction=0.0),
        "shap": make_redteam_result(flipped=False, edit_fraction=0.0),
        "lime": make_redteam_result(flipped=False, edit_fraction=0.0),
    }
    judge = JudgeAgent(plausibility_fn=fixed_plausibility_fn(score=5))
    state = judge.run(state)

    for verdict in state["judge"]["verdicts"].values():
        assert verdict["faithfulness_confidence"] == "faithful"
        assert verdict["divergent"] is False
        # not-flipped is exactly the inconclusive case a deeper retry exists
        # for -- "faithful" here is provisional until that retry confirms it.
        assert verdict["needs_deeper_redteam"] is True


def test_judge_needs_deeper_redteam_only_when_not_flipped():
    state = base_state()
    state["redteam"] = {
        "attention": make_redteam_result(flipped=False, edit_fraction=0.0),
        "shap": make_redteam_result(flipped=True, edit_fraction=0.1),
        "lime": make_redteam_result(flipped=True, edit_fraction=0.1),
    }
    judge = JudgeAgent(plausibility_fn=fixed_plausibility_fn(score=3))
    state = judge.run(state)

    assert state["judge"]["verdicts"]["attention"]["needs_deeper_redteam"] is True
    assert state["judge"]["verdicts"]["shap"]["needs_deeper_redteam"] is False
    assert state["judge"]["verdicts"]["lime"]["needs_deeper_redteam"] is False


def test_judge_reuses_cached_plausibility_without_recalling_fn():
    calls = []

    def counting_fn(prompt_text, target, words, values):
        calls.append(1)
        return {"score": 4, "reasoning": "stub"}

    state = base_state()
    state["redteam"] = {
        "attention": make_redteam_result(flipped=False, edit_fraction=0.0),
        "shap": make_redteam_result(flipped=False, edit_fraction=0.0),
        "lime": make_redteam_result(flipped=False, edit_fraction=0.0),
    }
    judge = JudgeAgent(plausibility_fn=counting_fn)
    state = judge.run(state)
    assert len(calls) == 3

    # Simulate the orchestrator's deeper retry: re-running the judge after
    # red-team results changed should not re-query plausibility.
    state["redteam"]["attention"] = make_redteam_result(flipped=True, edit_fraction=0.4)
    state = judge.run(state)
    assert len(calls) == 3


def test_orchestrator_end_to_end_on_tiny_moe_model():
    config = make_config()
    model = MoEGPT(config)
    orchestrator = Orchestrator(
        model,
        TOKENIZER,
        "cpu",
        config.block_size,
        shap_max_evals=20,
        lime_num_samples=20,
        redteam_top_n_neighbors=4,
        redteam_max_edits=2,
        deep_top_n_neighbors=4,
        deep_max_edits=3,
        judge_plausibility_fn=fixed_plausibility_fn(score=3),
    )

    state = orchestrator.run("Hello world, how are you?")

    assert set(state["judge"]["verdicts"].keys()) == {"attention", "shap", "lime"}
    assert isinstance(state["redteam_deep_retry"], list)
    assert "# RouteLens explanation report" in state["report_markdown"]
    # Any explainer flagged for retry should have been re-attacked with the
    # deeper budget -- its result should reflect the deep max_edits, not the
    # cheap one, whenever it still didn't flip.
    for name in state["redteam_deep_retry"]:
        rt = state["redteam"][name]
        assert rt.num_edits <= 3
