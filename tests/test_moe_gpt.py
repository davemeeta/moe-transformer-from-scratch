import math

import torch

from moe_transformer.config import ModelConfig
from moe_transformer.models import DenseGPT, MoEGPT


def make_model(**overrides):
    defaults = dict(
        vocab_size=50,
        n_embd=16,
        n_head=2,
        n_layer=3,
        block_size=8,
        dropout=0.0,
        num_experts=4,
        top_k=2,
        capacity_factor=100.0,
    )
    defaults.update(overrides)
    config = ModelConfig(**defaults)
    return MoEGPT(config), config


def test_output_shape():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss, aux = model(idx)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is None
    assert aux["ce_loss"] is None


def test_weight_tying():
    model, _ = make_model()
    assert model.lm_head.weight is model.token_emb.weight


def test_sequence_longer_than_block_size_raises():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
    try:
        model(idx)
        assert False, "expected an assertion error for sequence > block_size"
    except AssertionError:
        pass


def test_initial_loss_near_ln_vocab_size():
    config = ModelConfig(
        vocab_size=50257,
        n_embd=32,
        n_head=2,
        n_layer=2,
        block_size=16,
        dropout=0.0,
        num_experts=4,
        top_k=2,
    )
    model = MoEGPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (4, 16))
    targets = torch.randint(0, config.vocab_size, (4, 16))
    _, loss, aux = model(idx, targets)
    expected = math.log(config.vocab_size)
    # aux losses are tiny-weighted (0.01, 0.001) so shouldn't move this much
    assert abs(loss.item() - expected) < 0.5, (
        f"loss {loss.item():.3f} too far from ln(vocab_size)={expected:.3f}"
    )


def test_aux_losses_are_averaged_across_layers():
    model, config = make_model()
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        x = model.token_emb(idx)
        lb_losses, z_losses = [], []
        for block in model.blocks:
            x, moe_out = block(x)
            lb_losses.append(moe_out.load_balancing_loss)
            z_losses.append(moe_out.z_loss)
        expected_lb = torch.stack(lb_losses).mean()
        expected_z = torch.stack(z_losses).mean()

    _, _, aux = model(idx)
    assert torch.allclose(aux["load_balancing_loss"], expected_lb, atol=1e-5)
    assert torch.allclose(aux["z_loss"], expected_z, atol=1e-5)


def test_gradients_reach_every_expert_and_router_in_every_layer():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    _, loss, _ = model(idx, targets)
    loss.backward()

    for layer_idx, block in enumerate(model.blocks):
        assert block.moe.gate.weight.grad is not None, f"layer {layer_idx} gate has no grad"
        assert torch.isfinite(block.moe.gate.weight.grad).all()
        for e, expert in enumerate(block.moe.experts):
            grad = expert.down_proj.weight.grad
            assert grad is not None, f"layer {layer_idx} expert {e} has no grad"
            assert torch.isfinite(grad).all()


def test_overfits_a_tiny_batch():
    model, config = make_model()
    torch.manual_seed(0)
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))
    targets = torch.randint(0, config.vocab_size, (4, config.block_size))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    _, initial_loss, _ = model(idx, targets)

    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        _, loss, _ = model(idx, targets)
        loss.backward()
        optimizer.step()

    _, final_loss, _ = model(idx, targets)
    assert final_loss.item() < initial_loss.item() * 0.2


def test_generate_shape_and_valid_token_ids():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (1, 3))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8)
    assert torch.all(out >= 0) and torch.all(out < config.vocab_size)


def test_active_params_close_to_dense_baseline_but_total_is_larger():
    shared = dict(
        vocab_size=1000, n_embd=64, n_head=4, n_layer=2, block_size=16, dropout=0.0
    )
    dense = DenseGPT(ModelConfig(**shared))
    moe = MoEGPT(ModelConfig(**shared, num_experts=4, top_k=2))

    dense_params = dense.get_num_params(exclude_embedding=True)
    moe_active = moe.get_active_num_params(exclude_embedding=True)
    moe_total = moe.get_num_params(exclude_embedding=True)

    # active (non-embedding) params should be close to the dense baseline --
    # this is the "iso-FLOP" claim from config.moe_expert_hidden_dim.
    relative_diff = abs(moe_active - dense_params) / dense_params
    assert relative_diff < 0.15, (
        f"active params {moe_active} too far from dense {dense_params} "
        f"({relative_diff:.1%} relative difference)"
    )

    # total params should be meaningfully larger: more capacity, same active compute.
    assert moe_total > dense_params * 1.3
