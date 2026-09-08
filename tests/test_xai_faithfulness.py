import pytest
import torch

from moe_transformer.config import ModelConfig
from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.agents.explainer_agent import ExplainerAgent
from moe_transformer.xai.agents.faithfulness_agent import FaithfulnessAgent
from moe_transformer.xai.faithfulness.comp_suff import comprehensiveness, sufficiency
from moe_transformer.xai.faithfulness.deletion_insertion import (
    deletion_curve,
    insertion_curve,
)
from moe_transformer.xai.faithfulness.router_attribution_agreement import (
    router_attribution_agreement,
)
from moe_transformer.xai.faithfulness.utils import auc, make_bpe_eval_fn

TOKENIZER = Tokenizer()

# A synthetic "perfect explainer" fixture: eval_fn(kept) is the fraction of
# total true_importance retained, and `values` ranks tokens exactly the way
# true_importance does -- lets deletion/insertion/comp/suff be checked
# against hand-computed numbers without needing a real model forward pass.
TRUE_IMPORTANCE = [0.1, 0.9, 0.5]
VALUES = [0.1, 0.9, 0.5]


def synthetic_eval_fn(kept_positions: set[int]) -> float:
    if not kept_positions:
        return 0.0
    return sum(TRUE_IMPORTANCE[i] for i in kept_positions) / sum(TRUE_IMPORTANCE)


def test_deletion_curve_perfect_explainer_crashes_monotonically():
    fractions, probs = deletion_curve(VALUES, synthetic_eval_fn)
    assert fractions == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]
    assert probs[0] == pytest.approx(1.0)
    assert probs[-1] == pytest.approx(0.0)
    assert all(a >= b - 1e-9 for a, b in zip(probs, probs[1:])), "deletion should be non-increasing"


def test_insertion_curve_perfect_explainer_rises_monotonically():
    fractions, probs = insertion_curve(VALUES, synthetic_eval_fn)
    assert probs[0] == pytest.approx(0.0)
    assert probs[-1] == pytest.approx(1.0)
    assert all(a <= b + 1e-9 for a, b in zip(probs, probs[1:])), "insertion should be non-decreasing"


def test_perfect_explainer_deletion_auc_below_insertion_auc():
    del_fractions, del_probs = deletion_curve(VALUES, synthetic_eval_fn)
    ins_fractions, ins_probs = insertion_curve(VALUES, synthetic_eval_fn)
    assert auc(del_fractions, del_probs) < auc(ins_fractions, ins_probs)


def test_comprehensiveness_and_sufficiency_hand_computed():
    original_prob = synthetic_eval_fn({0, 1, 2})
    comp = comprehensiveness(VALUES, 1 / 3, original_prob, synthetic_eval_fn)
    suff = sufficiency(VALUES, 1 / 3, original_prob, synthetic_eval_fn)
    # top-1 by |value| is index 1 (0.9); comp removes it, suff keeps only it.
    assert comp == pytest.approx(1.0 - (0.1 + 0.5) / 1.5)
    assert suff == pytest.approx(1.0 - 0.9 / 1.5)


def make_router_trace(gate_weights: list[float]) -> dict:
    detail = [[(0, w)] for w in gate_weights]
    return {"detail": [detail]}


def test_router_attribution_agreement_perfect_correlation():
    gate_weights = [0.1, 0.9, 0.5]
    tau = router_attribution_agreement(gate_weights, make_router_trace(gate_weights))
    assert tau == pytest.approx(1.0)


def test_router_attribution_agreement_anti_correlation():
    attribution_values = [0.9, 0.5, 0.1]
    gate_weights = [0.1, 0.5, 0.9]
    tau = router_attribution_agreement(attribution_values, make_router_trace(gate_weights))
    assert tau == pytest.approx(-1.0)


def test_router_attribution_agreement_length_mismatch_raises():
    with pytest.raises(ValueError):
        router_attribution_agreement([0.1, 0.2], make_router_trace([0.1, 0.2, 0.3]))


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


def test_bpe_eval_fn_returns_valid_probability():
    config = make_config()
    model = MoEGPT(config)
    token_ids = TOKENIZER.encode("Hello world")
    eval_fn = make_bpe_eval_fn(model, TOKENIZER, "cpu", token_ids, target_token_id=5)
    prob = eval_fn(set(range(len(token_ids))))
    assert 0.0 <= prob <= 1.0
    empty_prob = eval_fn(set())
    assert 0.0 <= empty_prob <= 1.0


def _assert_valid_scores(scores: dict, expect_router_agreement: bool):
    assert 0.0 <= scores["deletion_auc"] <= 1.0
    assert 0.0 <= scores["insertion_auc"] <= 1.0
    assert -1.0 <= scores["comprehensiveness"] <= 1.0
    assert -1.0 <= scores["sufficiency"] <= 1.0
    assert 0.0 <= scores["original_prob"] <= 1.0
    if expect_router_agreement:
        assert -1.0 <= scores["router_attribution_agreement"] <= 1.0
    else:
        assert "router_attribution_agreement" not in scores


def test_faithfulness_agent_end_to_end_on_moe_model():
    config = make_config()
    model = MoEGPT(config)
    explainer = ExplainerAgent(
        model, TOKENIZER, "cpu", config.block_size, shap_max_evals=20, lime_num_samples=20
    )
    faithfulness = FaithfulnessAgent(model, TOKENIZER, "cpu", config.block_size)

    state = explainer.run({"text": "Hello world"})
    state = faithfulness.run(state)

    assert set(state["faithfulness"].keys()) == {"attention", "shap", "lime"}
    _assert_valid_scores(state["faithfulness"]["attention"], expect_router_agreement=True)
    _assert_valid_scores(state["faithfulness"]["shap"], expect_router_agreement=False)
    _assert_valid_scores(state["faithfulness"]["lime"], expect_router_agreement=False)


def test_faithfulness_agent_end_to_end_on_dense_model_has_no_router_agreement():
    config = make_config()
    model = DenseGPT(config)
    explainer = ExplainerAgent(
        model, TOKENIZER, "cpu", config.block_size, shap_max_evals=20, lime_num_samples=20
    )
    faithfulness = FaithfulnessAgent(model, TOKENIZER, "cpu", config.block_size)

    state = explainer.run({"text": "Hello world"})
    state = faithfulness.run(state)

    _assert_valid_scores(state["faithfulness"]["attention"], expect_router_agreement=False)
