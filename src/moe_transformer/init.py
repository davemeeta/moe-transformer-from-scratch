"""GPT-2-style weight init.

Default nn.Embedding init (std=1.0) makes initial cross-entropy loss much
higher than the expected ln(vocab_size) baseline. GPT-2 instead initializes
Linear and Embedding weights from N(0, 0.02) with zeroed Linear bias.
"""

from __future__ import annotations

import torch.nn as nn

STD = 0.02


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=STD)
