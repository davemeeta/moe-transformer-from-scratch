import torch

from moe_transformer.model.rope import RotaryEmbedding, apply_rotary_pos_emb


def test_position_zero_is_identity():
    head_dim = 8
    rope = RotaryEmbedding(head_dim, max_seq_len=16)
    x = torch.randn(1, 1, 16, head_dim)
    cos, sin = rope(seq_len=16)
    rotated = apply_rotary_pos_emb(x, cos, sin)
    assert torch.allclose(rotated[:, :, 0, :], x[:, :, 0, :], atol=1e-6)


def test_rotation_preserves_norm():
    head_dim = 8
    rope = RotaryEmbedding(head_dim, max_seq_len=16)
    x = torch.randn(2, 3, 16, head_dim)
    cos, sin = rope(seq_len=16)
    rotated = apply_rotary_pos_emb(x, cos, sin)
    assert torch.allclose(
        rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-4
    )


def test_different_positions_rotate_differently():
    head_dim = 8
    rope = RotaryEmbedding(head_dim, max_seq_len=16)
    x = torch.randn(1, 1, 16, head_dim)
    x = x.expand(1, 1, 16, head_dim).clone()
    x[:, :, :, :] = x[:, :, 0:1, :]  # identical vector at every position
    cos, sin = rope(seq_len=16)
    rotated = apply_rotary_pos_emb(x, cos, sin)
    assert not torch.allclose(rotated[:, :, 1, :], rotated[:, :, 5, :])
