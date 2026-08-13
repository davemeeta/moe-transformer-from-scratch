"""Checkpointing: model weights in safetensors, training state alongside.

Plain state_dict-based safetensors saving fails on this project's models
because the token embedding and lm_head share one tensor (see models.py's
weight tying) -- safetensors forbids writing the same storage under two
keys. safetensors.torch.save_model/load_model exist specifically to handle
this: they detect shared tensors, store one copy, and re-tie on load.

Optimizer state (nested tensors + non-tensor bookkeeping) doesn't fit
safetensors' flat-tensor-dict format, and it's never meant to be shared or
loaded by anyone but this same training run, so it's saved as a plain
torch.save sidecar rather than forcing it into safetensors too. The model
weights -- the actual artifact worth sharing or loading elsewhere -- are
what get the safetensors treatment.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_model, save_model

MODEL_FILENAME = "model.safetensors"
TRAIN_STATE_FILENAME = "train_state.pt"
CONFIG_FILENAME = "config.yaml"


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: DictConfig,
    checkpoint_dir: str | Path,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_model(model, str(checkpoint_dir / MODEL_FILENAME))
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step},
        checkpoint_dir / TRAIN_STATE_FILENAME,
    )
    OmegaConf.save(cfg, checkpoint_dir / CONFIG_FILENAME)


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> int:
    """Loads weights (and optimizer state, if given) in place. Returns the
    step the checkpoint was saved at, so training can resume from step+1."""
    checkpoint_dir = Path(checkpoint_dir)
    load_model(model, str(checkpoint_dir / MODEL_FILENAME), device=device)

    train_state = torch.load(
        checkpoint_dir / TRAIN_STATE_FILENAME, map_location=device
    )
    if optimizer is not None:
        optimizer.load_state_dict(train_state["optimizer"])
    return train_state["step"]
