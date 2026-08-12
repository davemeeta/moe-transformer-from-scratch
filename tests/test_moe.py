import torch

from moe_transformer.config import ModelConfig
from moe_transformer.model import MoELayer


def make_moe(**overrides):
    defaults = dict(
        vocab_size=50,
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=8,
        dropout=0.0,
        num_experts=4,
        top_k=2,
        capacity_factor=100.0,  # generous by default: no drops unless a test wants them
    )
    defaults.update(overrides)
    config = ModelConfig(**defaults)
    return MoELayer(config), config


def _eye_routed_moe(num_experts=4, top_k=1, capacity_factor=100.0):
    """A MoE layer whose gate weight is the identity matrix, so router
    logits equal the input vector exactly -- lets tests control routing
    outcomes precisely by choosing input vectors instead of fighting
    softmax/init randomness."""
    config = ModelConfig(
        vocab_size=50,
        n_embd=num_experts,
        n_head=1,
        n_layer=1,
        block_size=64,
        dropout=0.0,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
    )
    moe = MoELayer(config)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.eye(num_experts))
    return moe, config


def test_output_shape():
    moe, config = make_moe()
    x = torch.randn(2, 6, config.n_embd)
    out = moe(x)
    assert out.output.shape == x.shape


def test_topk_weights_renormalize_to_one():
    moe, config = make_moe()
    x_flat = torch.randn(30, config.n_embd)
    _, _, _, topk_weights = moe.route(x_flat)
    sums = topk_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_full_probs_sum_to_one_per_token():
    moe, config = make_moe()
    x_flat = torch.randn(30, config.n_embd)
    _, full_probs, _, _ = moe.route(x_flat)
    sums = full_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_dispatch_matches_manual_per_token_computation():
    torch.manual_seed(0)
    moe, config = make_moe(n_embd=6, num_experts=3, top_k=2)
    x = torch.randn(1, 5, config.n_embd)
    x_flat = x.reshape(-1, config.n_embd)

    out = moe(x)
    _, _, topk_idx, topk_weights = moe.route(x_flat)

    expected = torch.zeros_like(x_flat)
    for token in range(x_flat.size(0)):
        for slot in range(config.top_k):
            e = topk_idx[token, slot].item()
            w = topk_weights[token, slot]
            expected[token] += w * moe.experts[e](x_flat[token : token + 1]).squeeze(0)

    assert torch.allclose(out.output.reshape(-1, config.n_embd), expected, atol=1e-5)
    assert out.num_dropped_tokens == 0


def test_capacity_overflow_drops_tokens_in_position_order():
    num_experts = 4
    moe, config = _eye_routed_moe(num_experts=num_experts, top_k=1, capacity_factor=0.5)

    N = 16
    x = torch.zeros(1, N, num_experts)
    x[:, :, 0] = 1.0  # every token routes to expert 0

    out = moe(x)

    capacity = int(config.capacity_factor * N * config.top_k / config.num_experts)
    assert capacity == 2
    assert out.num_dropped_tokens == N - capacity
    assert out.expert_assignment_counts[0].item() == N
    assert out.expert_assignment_counts[1:].sum().item() == 0

    flat_out = out.output.reshape(N, num_experts)
    nonzero_rows = flat_out.abs().sum(dim=-1) > 1e-8
    assert nonzero_rows[:capacity].all(), "tokens within capacity should be processed"
    assert not nonzero_rows[capacity:].any(), "tokens beyond capacity should be dropped"


def test_partial_drop_keeps_contribution_from_non_full_expert():
    # top_k=2 over 3 experts. Every token's #1 choice is expert 0 (which will
    # overflow), but each token's #2 choice is spread across experts 1/2
    # (which won't overflow). Tokens beyond expert 0's capacity should still
    # receive expert 1/2's contribution -- not be zeroed out entirely.
    # capacity is shared across all experts (capacity_factor * N * top_k / num_experts),
    # so pick capacity_factor so expert 0's full load (N) exceeds it but each of
    # experts 1/2's half-load (N/2) doesn't: need N/2 <= capacity < N.
    num_experts = 3
    moe, config = _eye_routed_moe(num_experts=num_experts, top_k=2, capacity_factor=1.0)

    N = 12
    x = torch.zeros(1, N, num_experts)
    x[:, :, 0] = 5.0  # strong preference for expert 0 (top-1 for everyone)
    for j in range(N):
        other = 1 if j % 2 == 0 else 2
        x[0, j, other] = 1.0  # weaker #2 preference, alternating between experts 1 and 2

    out = moe(x)
    capacity = int(config.capacity_factor * N * config.top_k / config.num_experts)
    assert 0 < capacity < N  # expert 0 must overflow for this test to mean anything
    assert out.num_dropped_tokens > 0

    flat_out = out.output.reshape(N, num_experts)
    nonzero_rows = flat_out.abs().sum(dim=-1) > 1e-8
    # tokens beyond expert 0's capacity should NOT be all-zero: experts 1/2
    # still had room and should have contributed.
    assert nonzero_rows[capacity:].all()


def test_gradients_reach_router_via_load_balancing_loss():
    moe, config = make_moe()
    x = torch.randn(2, 4, config.n_embd)
    out = moe(x)
    out.load_balancing_loss.backward()
    assert moe.gate.weight.grad is not None
    assert torch.isfinite(moe.gate.weight.grad).all()
    assert moe.gate.weight.grad.abs().sum().item() > 0


def test_gradients_reach_router_via_z_loss():
    moe, config = make_moe()
    x = torch.randn(2, 4, config.n_embd)
    out = moe(x)
    out.z_loss.backward()
    assert moe.gate.weight.grad is not None
    assert torch.isfinite(moe.gate.weight.grad).all()
    assert moe.gate.weight.grad.abs().sum().item() > 0


def test_collapsed_router_scores_worse_than_balanced_on_load_balancing_loss():
    num_experts = 4
    scale = 5.0
    N = 40

    balanced_moe, _ = _eye_routed_moe(num_experts=num_experts, top_k=1)
    x_balanced = torch.zeros(1, N, num_experts)
    for j in range(N):
        x_balanced[0, j, j % num_experts] = scale
    balanced_out = balanced_moe(x_balanced)

    collapsed_moe, _ = _eye_routed_moe(num_experts=num_experts, top_k=1)
    x_collapsed = torch.zeros(1, N, num_experts)
    x_collapsed[:, :, 0] = scale
    collapsed_out = collapsed_moe(x_collapsed)

    balanced_loss = balanced_out.load_balancing_loss.item()
    collapsed_loss = collapsed_out.load_balancing_loss.item()

    assert collapsed_loss > balanced_loss
    assert balanced_loss < 1.5  # theoretical minimum is 1.0 (perfectly balanced)
    assert collapsed_loss > 2.5  # theoretical maximum is num_experts=4.0 (fully collapsed)


def test_z_loss_grows_with_logit_magnitude():
    num_experts = 4
    small_moe, _ = _eye_routed_moe(num_experts=num_experts, top_k=1)
    big_moe, _ = _eye_routed_moe(num_experts=num_experts, top_k=1)

    x_small = torch.ones(1, 8, num_experts) * 0.1
    x_big = torch.ones(1, 8, num_experts) * 10.0

    small_z = small_moe(x_small).z_loss.item()
    big_z = big_moe(x_big).z_loss.item()
    assert big_z > small_z
