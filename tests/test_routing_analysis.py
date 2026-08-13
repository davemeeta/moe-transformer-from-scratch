import torch

from moe_transformer.config import ModelConfig
from moe_transformer.model.moe import MoELayer, MoEOutput
from moe_transformer.routing_analysis import (
    accumulate_layer_counts,
    collapse_score,
    per_token_routing_table,
    utilization_from_counts,
)


def _fake_moe_output(expert_assignment_counts, topk_idx, topk_weights) -> MoEOutput:
    N = topk_idx.shape[0]
    C = 4  # arbitrary, unused by the analysis functions under test
    return MoEOutput(
        output=torch.zeros(N, C),
        load_balancing_loss=torch.tensor(0.0),
        z_loss=torch.tensor(0.0),
        num_dropped_tokens=0,
        expert_assignment_counts=expert_assignment_counts,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )


def test_collapse_score_zero_for_uniform_distribution():
    f_i = torch.full((4,), 0.25)
    assert collapse_score(f_i).item() < 1e-5


def test_collapse_score_one_for_fully_collapsed():
    f_i = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert abs(collapse_score(f_i).item() - 1.0) < 1e-5


def test_collapse_score_is_between_zero_and_one_for_partial_collapse():
    f_i = torch.tensor([0.7, 0.1, 0.1, 0.1])
    score = collapse_score(f_i).item()
    assert 0.0 < score < 1.0


def test_collapse_score_orders_balanced_below_partial_below_collapsed():
    balanced = collapse_score(torch.full((4,), 0.25)).item()
    partial = collapse_score(torch.tensor([0.55, 0.15, 0.15, 0.15])).item()
    collapsed = collapse_score(torch.tensor([1.0, 0.0, 0.0, 0.0])).item()
    assert balanced < partial < collapsed


def test_collapse_score_works_per_layer_batched():
    f_i = torch.stack([torch.full((4,), 0.25), torch.tensor([1.0, 0.0, 0.0, 0.0])])
    scores = collapse_score(f_i)
    assert scores.shape == (2,)
    assert scores[0].item() < 1e-5
    assert abs(scores[1].item() - 1.0) < 1e-5


def test_utilization_from_counts_sums_to_one_per_layer():
    counts = torch.tensor([[10, 5, 5, 0], [5, 5, 5, 5]])  # (n_layer=2, num_experts=4)
    total_tokens = 10
    top_k = 2  # sum(counts per layer) == total_tokens * top_k == 20
    f_i = utilization_from_counts(counts, total_tokens, top_k)
    sums = f_i.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-6)


def test_accumulate_layer_counts_sums_across_batches_and_layers():
    # 2 layers, 3 experts, top_k=2, batch of N=5 tokens each time
    batch1 = [
        _fake_moe_output(
            torch.tensor([2, 3, 0]), torch.zeros(5, 2, dtype=torch.long), torch.zeros(5, 2)
        ),
        _fake_moe_output(
            torch.tensor([1, 1, 3]), torch.zeros(5, 2, dtype=torch.long), torch.zeros(5, 2)
        ),
    ]
    batch2 = [
        _fake_moe_output(
            torch.tensor([1, 1, 1]), torch.zeros(5, 2, dtype=torch.long), torch.zeros(5, 2)
        ),
        _fake_moe_output(
            torch.tensor([0, 2, 3]), torch.zeros(5, 2, dtype=torch.long), torch.zeros(5, 2)
        ),
    ]

    counts, total_tokens, top_k = accumulate_layer_counts([batch1, batch2])

    assert total_tokens == 10  # 5 + 5
    assert top_k == 2
    assert torch.equal(counts[0], torch.tensor([3, 4, 1]))  # layer 0: [2,3,0] + [1,1,1]
    assert torch.equal(counts[1], torch.tensor([1, 3, 6]))  # layer 1: [1,1,3] + [0,2,3]


def test_per_token_routing_table_matches_direct_topk():
    config = ModelConfig(
        vocab_size=50, n_embd=4, n_head=1, n_layer=1, block_size=8, dropout=0.0,
        num_experts=4, top_k=2, capacity_factor=100.0,
    )
    moe = MoELayer(config)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.eye(4))

    x = torch.zeros(1, 3, 4)
    x[0, 0, 0] = 5.0  # token 0 strongly prefers expert 0
    x[0, 1, 1] = 5.0  # token 1 strongly prefers expert 1
    x[0, 2, 2] = 5.0  # token 2 strongly prefers expert 2

    moe_out = moe(x)
    tokens = ["a", "b", "c"]
    table = per_token_routing_table([moe_out], tokens)

    assert table["tokens"] == tokens
    assert table["top1_experts"].shape == (1, 3)
    assert table["top1_experts"][0].tolist() == [0, 1, 2]
    assert len(table["detail"]) == 1
    assert len(table["detail"][0]) == 3
    assert len(table["detail"][0][0]) == config.top_k  # each token has k (expert, weight) pairs
