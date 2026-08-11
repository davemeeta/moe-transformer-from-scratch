import pytest
import torch

from moe_transformer.data import TokenDataset, split_ids, write_bin


def test_split_ids_preserves_order_and_fraction():
    ids = list(range(100))
    train, val = split_ids(ids, val_fraction=0.1)
    assert train == list(range(90))
    assert val == list(range(90, 100))
    assert train + val == ids


def test_write_bin_and_dataset_shapes(tmp_path):
    ids = list(range(50))
    bin_path = tmp_path / "toy.bin"
    write_bin(ids, bin_path)

    block_size = 8
    ds = TokenDataset(bin_path, block_size=block_size)
    assert len(ds) == 50 - block_size

    x, y = ds[0]
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_dataset_targets_are_shifted_by_one(tmp_path):
    ids = list(range(50))
    bin_path = tmp_path / "toy.bin"
    write_bin(ids, bin_path)

    ds = TokenDataset(bin_path, block_size=8)
    idx = 5
    x, y = ds[idx]
    assert x.tolist() == ids[idx : idx + 8]
    assert y.tolist() == ids[idx + 1 : idx + 9]


def test_dataset_raises_when_corpus_smaller_than_block_size(tmp_path):
    ids = list(range(5))
    bin_path = tmp_path / "tiny.bin"
    write_bin(ids, bin_path)

    with pytest.raises(ValueError):
        TokenDataset(bin_path, block_size=8)
