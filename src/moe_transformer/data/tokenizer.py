"""GPT-2 BPE tokenizer, thin wrapper around tiktoken.

We use tiktoken's "gpt2" encoding directly rather than training our own BPE
merges — the vocabulary itself isn't the part of this project we're building
from scratch, the model internals are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tiktoken

ENCODING_NAME = "gpt2"


@dataclass
class Tokenizer:
    encoding_name: str = ENCODING_NAME

    def __post_init__(self) -> None:
        self._enc = tiktoken.get_encoding(self.encoding_name)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def save_meta(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(
                {"encoding_name": self.encoding_name, "vocab_size": self.vocab_size},
                indent=2,
            )
        )

    @classmethod
    def load_meta(cls, path: str | Path) -> "Tokenizer":
        meta = json.loads(Path(path).read_text())
        return cls(encoding_name=meta["encoding_name"])
