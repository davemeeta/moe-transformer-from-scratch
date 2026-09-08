import pytest
import torch

from moe_transformer.config import ModelConfig
from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.agents.explainer_agent import ExplainerAgent
from moe_transformer.xai.agents.redteam_agent import RedTeamAgent
from moe_transformer.xai.redteam.embedding_neighbors import nearest_neighbor_ids
from moe_transformer.xai.redteam.protected_positions import (
    protected_positions_bpe,
    protected_positions_from_words,
)
from moe_transformer.xai.redteam.search import necessity_attack
from moe_transformer.xai.utils import tokens_to_strs

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


def test_nearest_neighbor_ids_excludes_self_and_has_right_length():
    config = make_config()
    model = MoEGPT(config)
    token_id = 42
    neighbors = nearest_neighbor_ids(model.token_emb.weight, token_id, top_n=8)
    assert len(neighbors) == 8
    assert token_id not in neighbors
    assert all(0 <= n < config.vocab_size for n in neighbors)


def test_protected_positions_bpe_picks_top_k_by_magnitude():
    values = [0.1, 0.9, 0.5, -0.8]
    protected = protected_positions_bpe(values, k_fraction=0.5)
    assert protected == {1, 3}  # top-2 by |value|: 0.9 (idx 1), -0.8 (idx 3)


def test_protected_positions_from_words_matches_bpe_when_words_are_tokens():
    token_ids = TOKENIZER.encode("Hello world, how are you?")
    words = tokens_to_strs(TOKENIZER, token_ids)
    values = [float(i % 3) for i in range(len(words))]

    from_words = protected_positions_from_words(words, values, token_ids, TOKENIZER, k_fraction=0.3)
    from_bpe = protected_positions_bpe(values, k_fraction=0.3)
    assert from_words == from_bpe


def test_protected_positions_from_words_raises_on_reconstruction_mismatch():
    token_ids = TOKENIZER.encode("Hello world")
    with pytest.raises(ValueError):
        protected_positions_from_words(
            ["not", "the", "same", "text"], [0.1, 0.2, 0.3, 0.4], token_ids, TOKENIZER, k_fraction=0.5
        )


def test_necessity_attack_never_touches_protected_positions():
    config = make_config()
    model = MoEGPT(config)
    token_ids = TOKENIZER.encode("Hello world, how are you today?")
    protected = {0, 2, 4}

    result = necessity_attack(
        model, TOKENIZER, "cpu", token_ids, target_token_id=5, protected=protected,
        top_n_neighbors=4, max_edits=3,
    )

    edited_positions = {s["position"] for s in result.substitutions}
    assert edited_positions.isdisjoint(protected)
    assert result.num_edits == len(result.substitutions)
    assert result.num_edits <= 3
    assert 0.0 <= result.edit_fraction <= 1.0
    assert 0.0 <= result.original_prob <= 1.0
    assert 0.0 <= result.final_prob <= 1.0


def test_necessity_attack_with_everything_protected_does_nothing():
    config = make_config()
    model = MoEGPT(config)
    token_ids = TOKENIZER.encode("Hi there")
    protected = set(range(len(token_ids)))

    result = necessity_attack(
        model, TOKENIZER, "cpu", token_ids, target_token_id=5, protected=protected,
    )

    assert result.editable_positions == []
    assert result.num_edits == 0
    assert result.edit_fraction == 0.0
    assert result.flipped is False
    assert result.final_prob == pytest.approx(result.original_prob)


def test_necessity_attack_result_serializes_to_plain_dict():
    config = make_config()
    model = MoEGPT(config)
    token_ids = TOKENIZER.encode("Hello world, how are you?")
    result = necessity_attack(
        model, TOKENIZER, "cpu", token_ids, target_token_id=5, protected={0},
        top_n_neighbors=4, max_edits=2,
    )
    d = result.to_dict()
    assert set(d.keys()) == {
        "protected_positions", "editable_positions", "original_prob", "final_prob",
        "flipped", "num_edits", "edit_fraction", "substitutions",
    }
    assert isinstance(d["substitutions"], list)


def _assert_valid_result(rt):
    assert rt.num_edits == len(rt.substitutions)
    assert 0.0 <= rt.edit_fraction <= 1.0
    assert 0.0 <= rt.original_prob <= 1.0
    assert 0.0 <= rt.final_prob <= 1.0
    assert set(rt.editable_positions).isdisjoint(rt.protected_positions)


def test_redteam_agent_end_to_end_on_moe_model():
    config = make_config()
    model = MoEGPT(config)
    explainer = ExplainerAgent(
        model, TOKENIZER, "cpu", config.block_size, shap_max_evals=20, lime_num_samples=20
    )
    redteam = RedTeamAgent(model, TOKENIZER, "cpu", protected_k=0.3, top_n_neighbors=4, max_edits=2)

    state = explainer.run({"text": "Hello world, how are you?"})
    state = redteam.run(state)

    assert set(state["redteam"].keys()) == {"attention", "shap", "lime"}
    for rt in state["redteam"].values():
        _assert_valid_result(rt)


def test_redteam_agent_end_to_end_on_dense_model():
    config = make_config()
    model = DenseGPT(config)
    explainer = ExplainerAgent(
        model, TOKENIZER, "cpu", config.block_size, shap_max_evals=20, lime_num_samples=20
    )
    redteam = RedTeamAgent(model, TOKENIZER, "cpu", protected_k=0.3, top_n_neighbors=4, max_edits=2)

    state = explainer.run({"text": "Hello world, how are you?"})
    state = redteam.run(state)

    assert set(state["redteam"].keys()) == {"attention", "shap", "lime"}
    for rt in state["redteam"].values():
        _assert_valid_result(rt)
