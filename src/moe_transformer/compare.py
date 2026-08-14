"""Dense vs. MoE comparison: loss curves, total vs. active params, inference speed.

Usage:
    python -m moe_transformer.compare
    python -m moe_transformer.compare compare.dense_checkpoint=... compare.moe_checkpoint=...

Expects both models to already be trained (via moe_transformer.train) with
matched hyperparameters, so the comparison isolates the one variable that
matters: routing vs. no routing, not incidental differences in model size
or training budget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from moe_transformer.checkpoint import load_checkpoint
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.train import build_model_config


def load_metrics(run_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Reads a run's metrics.jsonl into (train_records, val_records).

    Uses ce_loss when present (MoE records log it separately from the total
    loss, which also includes weighted aux-loss terms) so both models'
    curves compare next-token prediction quality on equal footing.
    """
    train_records, val_records = [], []
    with open(Path(run_dir) / "metrics.jsonl") as f:
        for line in f:
            record = json.loads(line)
            entry = {"step": record["step"], "loss": record.get("ce_loss", record.get("loss"))}
            (train_records if record["split"] == "train" else val_records).append(entry)
    return train_records, val_records


def load_model_from_checkpoint(
    checkpoint_dir: str | Path, device: str
) -> tuple[DenseGPT | MoEGPT, object, str]:
    """Reconstructs a model from its own saved config.yaml (written alongside
    the checkpoint by moe_transformer.checkpoint.save_checkpoint), so dense
    and MoE don't need to be re-specified on compare.py's own CLI."""
    checkpoint_dir = Path(checkpoint_dir)
    saved_cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    model_config = build_model_config(saved_cfg.model)
    kind = saved_cfg.model.kind
    model = DenseGPT(model_config) if kind == "dense" else MoEGPT(model_config)
    load_checkpoint(checkpoint_dir, model, device=device)
    model.to(device)
    model.eval()
    return model, model_config, kind


@torch.no_grad()
def benchmark_forward_throughput(
    model: DenseGPT | MoEGPT,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    repeats: int,
    warmup: int,
) -> dict:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(warmup):
        model(x)
    t0 = time.time()
    for _ in range(repeats):
        model(x)
    elapsed = time.time() - t0
    return {
        "tokens_per_sec": (batch_size * seq_len * repeats) / elapsed,
        "ms_per_forward": elapsed / repeats * 1000,
    }


def plot_loss_comparison(dense_train, dense_val, moe_train, moe_val, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r["step"] for r in dense_train], [r["loss"] for r in dense_train],
        color="tab:blue", alpha=0.35, label="dense (train)",
    )
    ax.plot(
        [r["step"] for r in dense_val], [r["loss"] for r in dense_val],
        color="tab:blue", linewidth=2, marker="o", label="dense (val)",
    )
    ax.plot(
        [r["step"] for r in moe_train], [r["loss"] for r in moe_train],
        color="tab:orange", alpha=0.35, label="MoE (train)",
    )
    ax.plot(
        [r["step"] for r in moe_val], [r["loss"] for r in moe_val],
        color="tab:orange", linewidth=2, marker="o", label="MoE (val)",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Dense vs. MoE: loss for the same training budget")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cpu"  # only device both models can be fairly compared on -- see configs/compare/default.yaml
    print(f"device: {device} (fixed for both models -- see configs/compare/default.yaml)")

    dense_model, dense_config, dense_kind = load_model_from_checkpoint(cfg.compare.dense_checkpoint, device)
    moe_model, moe_config, moe_kind = load_model_from_checkpoint(cfg.compare.moe_checkpoint, device)
    assert dense_kind == "dense" and moe_kind == "moe", (
        "checkpoint kind mismatch -- check compare.dense_checkpoint/moe_checkpoint"
    )

    dense_train, dense_val = load_metrics(Path(cfg.compare.dense_checkpoint).parent)
    moe_train, moe_val = load_metrics(Path(cfg.compare.moe_checkpoint).parent)

    out_dir = Path(cfg.compare.out_dir)
    loss_plot_path = out_dir / "loss_comparison.png"
    plot_loss_comparison(dense_train, dense_val, moe_train, moe_val, loss_plot_path)
    print(f"wrote {loss_plot_path}")

    print("\nbenchmarking forward-pass throughput (no KV-cache; raw forward-pass cost)...")
    dense_bench = benchmark_forward_throughput(
        dense_model, dense_config.vocab_size, cfg.compare.benchmark_batch_size,
        min(cfg.compare.benchmark_seq_len, dense_config.block_size), device,
        cfg.compare.benchmark_repeats, cfg.compare.benchmark_warmup,
    )
    moe_bench = benchmark_forward_throughput(
        moe_model, moe_config.vocab_size, cfg.compare.benchmark_batch_size,
        min(cfg.compare.benchmark_seq_len, moe_config.block_size), device,
        cfg.compare.benchmark_repeats, cfg.compare.benchmark_warmup,
    )

    summary = {
        "dense": {
            "total_params": dense_model.get_num_params(),
            "final_train_loss": dense_train[-1]["loss"] if dense_train else None,
            "final_val_loss": dense_val[-1]["loss"] if dense_val else None,
            "inference": dense_bench,
        },
        "moe": {
            "total_params": moe_model.get_num_params(),
            "active_params": moe_model.get_active_num_params(),
            "final_train_loss": moe_train[-1]["loss"] if moe_train else None,
            "final_val_loss": moe_val[-1]["loss"] if moe_val else None,
            "inference": moe_bench,
        },
        "notes": [
            "Both models benchmarked on CPU: MPS was measured to be pathologically "
            "slow (and degrading step over step) for MoE's dynamic-shape capacity "
            "dispatch, so CPU is the only device both can be fairly compared on "
            "(see moe_transformer/model/moe.py's module docstring).",
            "Inference benchmark is single-batch forward-pass throughput with no "
            "KV-cache -- this project's generate() recomputes the full sequence "
            "each step, so this measures raw forward-pass cost, not an optimized "
            "autoregressive decoding path.",
            "loss values are cross-entropy only (excludes MoE's weighted "
            "load-balancing/z-loss terms), so the comparison reflects next-token "
            "prediction quality on equal footing.",
        ],
    }

    summary_path = out_dir / "comparison_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")

    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
