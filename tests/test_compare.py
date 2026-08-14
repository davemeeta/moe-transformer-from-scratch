import json

import torch
from omegaconf import OmegaConf

from moe_transformer.checkpoint import save_checkpoint
from moe_transformer.compare import (
    benchmark_forward_throughput,
    load_metrics,
    load_model_from_checkpoint,
)
from moe_transformer.config import ModelConfig
from moe_transformer.models import DenseGPT, MoEGPT


def test_load_metrics_splits_train_and_val(tmp_path):
    records = [
        {"step": 0, "split": "train", "loss": 10.8, "ce_loss": 10.8},
        {"step": 0, "split": "val", "loss": 10.9},  # old-format val record, no ce_loss
        {"step": 10, "split": "train", "loss": 9.5, "ce_loss": 9.4},
        {"step": 10, "split": "val", "loss": 9.6, "ce_loss": 9.55},
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records))

    train, val = load_metrics(tmp_path)
    assert [r["step"] for r in train] == [0, 10]
    assert [r["step"] for r in val] == [0, 10]

    # train records prefer ce_loss when present
    assert train[0]["loss"] == 10.8
    assert train[1]["loss"] == 9.4
    # val record without ce_loss falls back to loss; with ce_loss prefers it
    assert val[0]["loss"] == 10.9
    assert val[1]["loss"] == 9.55


def test_load_model_from_checkpoint_roundtrips_dense(tmp_path):
    config = ModelConfig(vocab_size=50, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.0)
    model = DenseGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    cfg = OmegaConf.create({"model": {"kind": "dense", **vars(config)}})
    save_checkpoint(model, optimizer, step=1, cfg=cfg, checkpoint_dir=tmp_path / "ckpt")

    loaded_model, loaded_config, kind = load_model_from_checkpoint(tmp_path / "ckpt", "cpu")
    assert kind == "dense"
    assert isinstance(loaded_model, DenseGPT)
    assert loaded_config.n_embd == config.n_embd
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
        assert torch.allclose(p1, p2), f"mismatch in {n1}"


def test_load_model_from_checkpoint_roundtrips_moe(tmp_path):
    config = ModelConfig(
        vocab_size=50, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.0,
        num_experts=3, top_k=2,
    )
    model = MoEGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    cfg = OmegaConf.create({"model": {"kind": "moe", **vars(config)}})
    save_checkpoint(model, optimizer, step=1, cfg=cfg, checkpoint_dir=tmp_path / "ckpt")

    loaded_model, loaded_config, kind = load_model_from_checkpoint(tmp_path / "ckpt", "cpu")
    assert kind == "moe"
    assert isinstance(loaded_model, MoEGPT)
    assert loaded_config.num_experts == config.num_experts


def test_benchmark_forward_throughput_returns_sane_values():
    config = ModelConfig(vocab_size=50, n_embd=16, n_head=2, n_layer=1, block_size=16, dropout=0.0)
    model = DenseGPT(config)
    model.eval()

    result = benchmark_forward_throughput(
        model, vocab_size=50, batch_size=2, seq_len=8, device="cpu", repeats=3, warmup=1
    )
    assert result["tokens_per_sec"] > 0
    assert result["ms_per_forward"] > 0
