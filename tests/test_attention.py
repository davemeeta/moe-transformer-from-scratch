import torch

from moe_transformer.config import ModelConfig
from moe_transformer.model import CausalSelfAttention


def make_attn(**overrides):
    config = ModelConfig(
        vocab_size=100, n_embd=16, n_head=4, n_layer=1, block_size=8, dropout=0.0, **overrides
    )
    return CausalSelfAttention(config), config


def test_output_shape():
    attn, config = make_attn()
    x = torch.randn(2, 8, config.n_embd)
    out = attn(x)
    assert out.shape == x.shape


def test_causality_future_tokens_dont_affect_past_outputs():
    attn, config = make_attn()
    attn.eval()
    x = torch.randn(1, 8, config.n_embd)

    out_a = attn(x)

    x_modified = x.clone()
    x_modified[:, 5:, :] = torch.randn_like(x_modified[:, 5:, :])  # change future
    out_b = attn(x_modified)

    assert torch.allclose(out_a[:, :5, :], out_b[:, :5, :], atol=1e-5)
    assert not torch.allclose(out_a[:, 5:, :], out_b[:, 5:, :])


def test_attention_weights_are_causal_and_normalized():
    attn, config = make_attn()
    attn.eval()
    x = torch.randn(1, 8, config.n_embd)
    _, attn_weights = attn(x, return_attn_weights=True)

    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    future_mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    future_weights = attn_weights[0, :, future_mask]
    assert torch.allclose(future_weights, torch.zeros_like(future_weights))
