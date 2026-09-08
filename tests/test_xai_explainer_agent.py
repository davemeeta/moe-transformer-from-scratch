import torch

from moe_transformer.config import ModelConfig
from moe_transformer.data.tokenizer import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.xai.agents.explainer_agent import ExplainerAgent
from moe_transformer.xai.explainers.attention import (
    attention_rollout,
    capture_attention_weights,
)
from moe_transformer.xai.explainers.router_trace import get_router_trace

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


def test_capture_attention_weights_does_not_change_model_output():
    config = make_config()
    model = MoEGPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 6))

    with torch.no_grad():
        baseline_logits, _, _ = model(idx)

    with capture_attention_weights(model) as captured:
        with torch.no_grad():
            patched_logits, _, _ = model(idx)

    assert torch.allclose(baseline_logits, patched_logits)
    assert len(captured) == config.n_layer
    for weights in captured:
        assert weights.shape == (1, config.n_head, 6, 6)


def test_attention_rollout_shape_and_normalization():
    config = make_config()
    model = MoEGPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 6))

    with capture_attention_weights(model) as captured:
        with torch.no_grad():
            model(idx)

    importance = attention_rollout(captured)
    assert importance.shape == (6,)
    assert torch.all(importance >= 0)
    assert abs(importance.sum().item() - 1.0) < 1e-5


def test_router_trace_none_for_dense_model():
    config = make_config()
    model = DenseGPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 5))
    assert get_router_trace(model, idx, ["a"] * 5) is None


def test_router_trace_shape_for_moe_model():
    config = make_config()
    model = MoEGPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 5))
    trace = get_router_trace(model, idx, ["a"] * 5)
    assert trace is not None
    assert trace["top1_experts"].shape == (config.n_layer, 5)
    assert len(trace["detail"]) == config.n_layer


def test_explainer_agent_run_on_moe_model():
    config = make_config()
    model = MoEGPT(config)
    agent = ExplainerAgent(
        model, TOKENIZER, "cpu", config.block_size, shap_max_evals=20, lime_num_samples=20
    )

    result = agent.run({"text": "Hello world"})

    n_tokens = len(result["tokens"])
    assert n_tokens == len(result["token_ids"]) == len(result["attention_importance"])
    assert 0 <= result["target_token_id"] < config.vocab_size
    assert isinstance(result["target_token_str"], str)
    assert len(result["shap_words"]) == len(result["shap_values"])
    assert len(result["lime_words"]) == len(result["lime_values"])
    assert result["router_trace"]["top1_experts"].shape == (config.n_layer, n_tokens)


def test_explainer_agent_run_on_dense_model_has_no_router_trace():
    config = make_config()
    model = DenseGPT(config)
    agent = ExplainerAgent(
        model, TOKENIZER, "cpu", config.block_size, shap_max_evals=20, lime_num_samples=20
    )

    result = agent.run({"text": "Hello world"})

    assert result["router_trace"] is None
    assert len(result["shap_words"]) == len(result["shap_values"])


def test_explainer_agent_respects_pinned_target_token():
    config = make_config()
    model = MoEGPT(config)
    agent = ExplainerAgent(
        model, TOKENIZER, "cpu", config.block_size, shap_max_evals=20, lime_num_samples=20
    )
    pinned_id = 42

    result = agent.run({"text": "Hello world", "target_token_id": pinned_id})

    assert result["target_token_id"] == pinned_id
    assert result["target_token_str"] == TOKENIZER.decode([pinned_id])
