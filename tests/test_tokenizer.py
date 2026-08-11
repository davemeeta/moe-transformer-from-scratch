from moe_transformer.data import Tokenizer


def test_roundtrip():
    tok = Tokenizer()
    text = "Hello, world! This is a test of the GPT-2 BPE tokenizer."
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text


def test_vocab_size():
    tok = Tokenizer()
    assert tok.vocab_size == 50257


def test_ids_within_vocab():
    tok = Tokenizer()
    ids = tok.encode("The quick brown fox jumps over the lazy dog.")
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_save_load_meta(tmp_path):
    tok = Tokenizer()
    meta_path = tmp_path / "tokenizer_meta.json"
    tok.save_meta(meta_path)

    loaded = Tokenizer.load_meta(meta_path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("round trip check") == tok.encode("round trip check")
