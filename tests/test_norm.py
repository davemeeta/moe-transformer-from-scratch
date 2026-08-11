import torch

from moe_transformer.model import RMSNorm


def test_shape_preserved():
    norm = RMSNorm(dim=16)
    x = torch.randn(2, 5, 16)
    assert norm(x).shape == x.shape


def test_unit_rms_when_weight_is_one():
    norm = RMSNorm(dim=32)
    x = torch.randn(4, 10, 32) * 5.0
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_zero_input_is_safe():
    norm = RMSNorm(dim=8)
    x = torch.zeros(1, 1, 8)
    out = norm(x)
    assert torch.isfinite(out).all()
