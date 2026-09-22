import numpy as np
import pytest

from moe_transformer.xai.agents.judge_agent import JudgeAgent
from moe_transformer.xai.redteam.search import RedTeamResult
from moe_transformer.xai.run_batch_eval import (
    aggregate,
    bootstrap_ci,
    sample_prompts,
    summarize,
)


def write_corpus(tmp_path, n_lines=40):
    lines = [f"This is a sample corpus line number {i} for testing." for i in range(n_lines)]
    lines += ["too short", "", "This is a sample corpus line number 3 for testing."]  # short/blank/dup
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(lines))
    return path


def test_sample_prompts_is_deterministic_for_a_seed(tmp_path):
    corpus = write_corpus(tmp_path)
    assert sample_prompts(corpus, 10, seed=1) == sample_prompts(corpus, 10, seed=1)


def test_sample_prompts_differs_across_seeds(tmp_path):
    corpus = write_corpus(tmp_path)
    assert sample_prompts(corpus, 10, seed=1) != sample_prompts(corpus, 10, seed=2)


def test_sample_prompts_are_unique_and_length_filtered(tmp_path):
    corpus = write_corpus(tmp_path)
    prompts = sample_prompts(corpus, 40, seed=0)
    assert len(prompts) == len(set(prompts))
    assert all(20 <= len(p) <= 100 for p in prompts)


def test_sample_prompts_returns_everything_when_asked_for_more_than_exist(tmp_path):
    corpus = write_corpus(tmp_path, n_lines=5)
    assert len(sample_prompts(corpus, 1000, seed=0)) == 5


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_more_data():
    rng = np.random.default_rng(0)
    small = rng.normal(0.5, 0.2, size=10).tolist()
    large = rng.normal(0.5, 0.2, size=400).tolist()

    low_s, high_s = bootstrap_ci(small, np.random.default_rng(1))
    low_l, high_l = bootstrap_ci(large, np.random.default_rng(1))

    assert low_s <= np.mean(small) <= high_s
    assert low_l <= np.mean(large) <= high_l
    assert (high_l - low_l) < (high_s - low_s)


def test_summarize_of_constant_values_has_zero_width_interval():
    s = summarize([1.0] * 20, np.random.default_rng(0))
    assert s["mean"] == 1.0
    assert s["ci_low"] == s["ci_high"] == 1.0
    assert s["std"] == 0.0


def make_record(flipped, plausibility=3, judge_failed=False, router_tau=None):
    row = {
        "deletion_auc": 0.1, "insertion_auc": 0.9, "comprehensiveness": 0.5,
        "sufficiency": 0.2, "router_tau": router_tau, "edit_fraction": 0.3,
        "flipped": flipped, "plausibility": plausibility,
        "judge_failed": judge_failed, "divergent": False,
    }
    return {"prompt": "p", "target_token": "x", "deep_retry": [],
            "explainers": {"attention": dict(row), "shap": dict(row), "lime": dict(row)}}


def test_aggregate_computes_flip_rate_over_prompts():
    records = [make_record(True), make_record(True), make_record(False), make_record(False)]
    agg = aggregate(records, seed=0)
    assert agg["lime"]["flip_rate"]["mean"] == pytest.approx(0.5)
    assert agg["lime"]["n_prompts"] == 4


def test_aggregate_excludes_failed_judge_ratings_from_plausibility():
    records = [
        make_record(True, plausibility=5),
        make_record(True, plausibility=5),
        make_record(True, plausibility=3, judge_failed=True),  # neutral default, must not count
    ]
    agg = aggregate(records, seed=0)
    assert agg["lime"]["n_judge_failures"] == 1
    assert agg["lime"]["plausibility"]["mean"] == pytest.approx(5.0)
    assert agg["lime"]["plausibility"]["n"] == 2


def test_aggregate_includes_router_tau_only_when_present():
    with_tau = aggregate([make_record(False, router_tau=0.2), make_record(False, router_tau=0.4)], seed=0)
    without_tau = aggregate([make_record(False), make_record(False)], seed=0)
    assert with_tau["attention"]["router_tau"]["mean"] == pytest.approx(0.3)
    assert "router_tau" not in without_tau["attention"]


def test_judge_marks_failed_ratings_and_never_flags_them_divergent():
    def failing_fn(prompt_text, target, words, values):
        return {"score": 3, "reasoning": "judge unavailable", "failed": True}

    def unfaithful_result():
        return RedTeamResult(
            protected_positions=[0], editable_positions=[1], original_prob=0.5,
            final_prob=0.1, flipped=True, num_edits=1, edit_fraction=0.05, substitutions=[],
        )

    state = {
        "text": "Hello world", "target_token_str": "!", "tokens": ["Hello", " world"],
        "attention_importance": [0.9, 0.1], "shap_words": ["Hello", " world"],
        "shap_values": [0.8, 0.2], "lime_words": ["Hello", " world"], "lime_values": [0.7, 0.3],
        "redteam": {n: unfaithful_result() for n in ("attention", "shap", "lime")},
    }
    state = JudgeAgent(plausibility_fn=failing_fn).run(state)
    for verdict in state["judge"]["verdicts"].values():
        assert verdict["judge_failed"] is True
        assert verdict["divergent"] is False
