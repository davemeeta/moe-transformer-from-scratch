import torch
from omegaconf import OmegaConf

from moe_transformer.checkpoint import load_checkpoint, save_checkpoint
from moe_transformer.config import ModelConfig
from moe_transformer.models import DenseGPT, MoEGPT


def test_dense_checkpoint_roundtrip_preserves_tied_weights(tmp_path):
    config = ModelConfig(vocab_size=50, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.0)
    model = DenseGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    loss.backward()
    optimizer.step()

    cfg = OmegaConf.create({"note": "test checkpoint"})
    save_checkpoint(model, optimizer, step=7, cfg=cfg, checkpoint_dir=tmp_path / "ckpt")

    fresh_model = DenseGPT(config)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    loaded_step = load_checkpoint(tmp_path / "ckpt", fresh_model, fresh_optimizer)

    assert loaded_step == 7
    assert fresh_model.lm_head.weight is fresh_model.token_emb.weight  # tying survives reload
    for (name, orig_param), (_, loaded_param) in zip(
        model.named_parameters(), fresh_model.named_parameters()
    ):
        assert torch.allclose(orig_param, loaded_param), f"mismatch in {name}"


def test_moe_checkpoint_roundtrip(tmp_path):
    config = ModelConfig(
        vocab_size=50, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.0,
        num_experts=3, top_k=2,
    )
    model = MoEGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    cfg = OmegaConf.create({"note": "test checkpoint"})
    save_checkpoint(model, optimizer, step=3, cfg=cfg, checkpoint_dir=tmp_path / "ckpt")

    fresh_model = MoEGPT(config)
    loaded_step = load_checkpoint(tmp_path / "ckpt", fresh_model)

    assert loaded_step == 3
    for (name, orig_param), (_, loaded_param) in zip(
        model.named_parameters(), fresh_model.named_parameters()
    ):
        assert torch.allclose(orig_param, loaded_param), f"mismatch in {name}"


def test_loaded_model_produces_identical_output():
    config = ModelConfig(vocab_size=50, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.0)
    model = DenseGPT(config)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = OmegaConf.create({})
        save_checkpoint(model, optimizer, step=1, cfg=cfg, checkpoint_dir=d)

        fresh_model = DenseGPT(config)
        fresh_model.eval()
        load_checkpoint(d, fresh_model)

        idx = torch.randint(0, config.vocab_size, (2, 8))
        logits_orig, _ = model(idx)
        logits_loaded, _ = fresh_model(idx)
        assert torch.allclose(logits_orig, logits_loaded)


def test_optimizer_state_survives_roundtrip(tmp_path):
    config = ModelConfig(vocab_size=50, n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0)
    model = DenseGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    loss.backward()
    optimizer.step()  # populate AdamW's internal exp_avg/exp_avg_sq state

    cfg = OmegaConf.create({})
    save_checkpoint(model, optimizer, step=1, cfg=cfg, checkpoint_dir=tmp_path / "ckpt")

    fresh_model = DenseGPT(config)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    load_checkpoint(tmp_path / "ckpt", fresh_model, fresh_optimizer)

    assert len(fresh_optimizer.state) == len(optimizer.state)
