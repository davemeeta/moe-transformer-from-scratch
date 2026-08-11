import torch

from moe_transformer.config import ModelConfig, swiglu_hidden_dim
from moe_transformer.model import FeedForward


def test_hidden_dim_formula():
    assert swiglu_hidden_dim(256) == 768
    assert swiglu_hidden_dim(256) % 256 == 0


def test_output_shape():
    config = ModelConfig(vocab_size=100, n_embd=16, n_head=2, n_layer=1, dropout=0.0)
    ffn = FeedForward(config)
    x = torch.randn(2, 5, config.n_embd)
    assert ffn(x).shape == x.shape
