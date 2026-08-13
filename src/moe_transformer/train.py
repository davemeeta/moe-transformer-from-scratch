"""Training loop for DenseGPT and MoEGPT, driven by Hydra config.

Usage:
    python -m moe_transformer.train model.kind=dense
    python -m moe_transformer.train model.kind=moe
    python -m moe_transformer.train training.max_steps=5000 model.num_experts=8 model.top_k=2
    python -m moe_transformer.train training.resume_from=checkpoints/dense/step_000500
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from moe_transformer.checkpoint import load_checkpoint, save_checkpoint
from moe_transformer.config import ModelConfig
from moe_transformer.data import TokenDataset
from moe_transformer.models import DenseGPT, MoEGPT


def resolve_device(requested: str, model_kind: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if model_kind == "dense" and torch.backends.mps.is_available():
        return "mps"
    # MoE's dynamic-shape capacity dispatch was measured to be pathologically
    # slow (and degrading step over step) on MPS -- see moe.py's docstring.
    return "cpu"


def build_model_config(model_cfg: DictConfig) -> ModelConfig:
    fields = OmegaConf.to_container(model_cfg, resolve=True)
    fields.pop("kind")
    return ModelConfig(**fields)


def get_lr(step: int, training_cfg: DictConfig) -> float:
    lr = training_cfg.learning_rate
    min_lr = lr * training_cfg.min_lr_ratio
    warmup = training_cfg.warmup_steps
    max_steps = training_cfg.max_steps

    if step < warmup:
        return lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def get_batch(
    dataset: TokenDataset, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    idxs = torch.randint(0, len(dataset), (batch_size,))
    xs, ys = zip(*[dataset[i] for i in idxs])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def forward_step(
    model: DenseGPT | MoEGPT, kind: str, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    if kind == "moe":
        _, loss, aux = model(x, y)
        log = {
            "loss": loss.item(),
            "ce_loss": aux["ce_loss"].item(),
            "lb_loss": aux["load_balancing_loss"].item(),
            "z_loss": aux["z_loss"].item(),
            "dropped_tokens": aux["num_dropped_tokens"],
        }
    else:
        _, loss = model(x, y)
        log = {"loss": loss.item()}
    return loss, log


@torch.no_grad()
def estimate_loss(
    model: DenseGPT | MoEGPT,
    kind: str,
    dataset: TokenDataset,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, device)
        loss, _ = forward_step(model, kind, x, y)
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    model_config = build_model_config(cfg.model)
    device = resolve_device(cfg.training.device, cfg.model.kind)
    print(f"device: {device}  model.kind: {cfg.model.kind}")

    if cfg.model.kind == "dense":
        model = DenseGPT(model_config)
    elif cfg.model.kind == "moe":
        model = MoEGPT(model_config)
    else:
        raise ValueError(f"unknown model.kind: {cfg.model.kind!r}, expected 'dense' or 'moe'")
    model.to(device)
    print(f"params: {model.get_num_params():,}")

    train_ds = TokenDataset(cfg.data.train_bin, block_size=model_config.block_size)
    val_ds = TokenDataset(cfg.data.val_bin, block_size=model_config.block_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    checkpoint_dir = Path(cfg.training.out_dir) / cfg.training.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = open(checkpoint_dir / "metrics.jsonl", "a")

    start_step = 0
    if cfg.training.resume_from:
        start_step = (
            load_checkpoint(cfg.training.resume_from, model, optimizer, device=device) + 1
        )
        print(f"resumed from {cfg.training.resume_from} at step {start_step}")

    t0 = time.time()
    for step in range(start_step, cfg.training.max_steps):
        lr = get_lr(step, cfg.training)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_ds, cfg.training.batch_size, device)
        loss, log = forward_step(model, cfg.model.kind, x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.training.grad_clip and cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        if step % cfg.training.log_interval == 0:
            elapsed = time.time() - t0
            ms_per_step = elapsed / (step - start_step + 1) * 1000
            metrics_str = "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in log.items()
            )
            print(f"step {step:5d}  lr {lr:.2e}  {metrics_str}  ({ms_per_step:.0f} ms/step)")
            metrics_file.write(json.dumps({"step": step, "split": "train", "lr": lr, **log}) + "\n")
            metrics_file.flush()

        if step % cfg.training.eval_interval == 0 and step > start_step:
            val_loss = estimate_loss(
                model, cfg.model.kind, val_ds, cfg.training.batch_size, cfg.training.eval_iters, device
            )
            print(f"step {step:5d}  val_loss {val_loss:.4f}")
            metrics_file.write(json.dumps({"step": step, "split": "val", "loss": val_loss}) + "\n")
            metrics_file.flush()

        if step % cfg.training.checkpoint_interval == 0 and step > start_step:
            step_dir = checkpoint_dir / f"step_{step:06d}"
            save_checkpoint(model, optimizer, step, cfg, step_dir)
            print(f"saved checkpoint to {step_dir}")

    final_dir = checkpoint_dir / f"step_{cfg.training.max_steps:06d}"
    save_checkpoint(model, optimizer, cfg.training.max_steps, cfg, final_dir)
    print(f"saved final checkpoint to {final_dir}")
    metrics_file.close()


if __name__ == "__main__":
    main()
