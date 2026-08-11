#!/usr/bin/env python
"""Tokenize a raw text corpus and write train/val token-id binaries.

Usage:
    python scripts/prepare_data.py --input data/raw/tinyshakespeare.txt --output-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moe_transformer.data import Tokenizer, split_ids, write_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw .txt corpus")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    text = args.input.read_text(encoding="utf-8")
    tokenizer = Tokenizer()
    ids = tokenizer.encode(text)

    train_ids, val_ids = split_ids(ids, val_fraction=args.val_fraction)

    write_bin(train_ids, args.output_dir / "train.bin")
    write_bin(val_ids, args.output_dir / "val.bin")
    tokenizer.save_meta(args.output_dir / "tokenizer_meta.json")

    print(f"input chars:   {len(text):,}")
    print(f"total tokens:  {len(ids):,}")
    print(f"train tokens:  {len(train_ids):,}")
    print(f"val tokens:    {len(val_ids):,}")
    print(f"vocab size:    {tokenizer.vocab_size:,}")
    print(f"wrote:         {args.output_dir}/{{train.bin,val.bin,tokenizer_meta.json}}")


if __name__ == "__main__":
    main()
