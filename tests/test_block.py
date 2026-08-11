import torch

from moe_transformer.config import ModelConfig
from moe_transformer.model import TransformerBlock


def make_block():
    config = ModelConfig(
        vocab_size=100, n_embd=16, n_head=4, n_layer=1, block_size=8, dropout=0.0
    )
    return TransformerBlock(config), config


def test_output_shape():
    block, config = make_block()
    x = torch.randn(2, 8, config.n_embd)
    assert block(x).shape == x.shape


def test_gradients_reach_all_parameters():
    block, config = make_block()
    x = torch.randn(2, 8, config.n_embd, requires_grad=True)
    out = block(x)
    out.sum().backward()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"no grad reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"


def test_deterministic_in_eval_mode():
    block, config = make_block()
    block.eval()
    x = torch.randn(1, 8, config.n_embd)
    out_a = block(x)
    out_b = block(x)
    assert torch.allclose(out_a, out_b)
