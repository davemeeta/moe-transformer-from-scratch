import math

import torch

from moe_transformer.config import ModelConfig
from moe_transformer.models import DenseGPT


def make_model(**overrides):
    config = ModelConfig(
        vocab_size=50,
        n_embd=32,
        n_head=2,
        n_layer=2,
        block_size=8,
        dropout=0.0,
        **overrides,
    )
    return DenseGPT(config), config


def test_output_shape():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is None


def test_loss_computed_when_targets_given():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    assert loss is not None
    assert loss.ndim == 0


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
    # Default GPT-2-style init (std=0.02) should keep the model's initial
    # predictions close to uniform, so cross-entropy loss should start near
    # ln(vocab_size) rather than the much higher value you'd get from
    # default nn.Embedding init (std=1.0).
    config = ModelConfig(
        vocab_size=50257, n_embd=32, n_head=2, n_layer=2, block_size=16, dropout=0.0
    )
    model = DenseGPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (4, 16))
    targets = torch.randint(0, config.vocab_size, (4, 16))
    _, loss = model(idx, targets)
    expected = math.log(config.vocab_size)
    assert abs(loss.item() - expected) < 0.3, (
        f"loss {loss.item():.3f} too far from ln(vocab_size)={expected:.3f}"
    )


def test_get_num_params_excludes_embedding_when_asked():
    model, config = make_model()
    total = model.get_num_params(exclude_embedding=False)
    non_embedding = model.get_num_params(exclude_embedding=True)
    assert total - non_embedding == config.vocab_size * config.n_embd
    assert non_embedding < total


def test_overfits_a_tiny_batch():
    model, config = make_model()
    torch.manual_seed(0)
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))
    targets = torch.randint(0, config.vocab_size, (4, config.block_size))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    _, initial_loss = model(idx, targets)

    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()

    _, final_loss = model(idx, targets)
    assert final_loss.item() < initial_loss.item() * 0.1


def test_generate_shape_and_valid_token_ids():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (1, 3))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8)
    assert torch.all(out >= 0) and torch.all(out < config.vocab_size)
