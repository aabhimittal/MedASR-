from medasr.tokenizer import BLANK_TOKEN, CharTokenizer


def test_build_reserves_blank_at_zero():
    tok = CharTokenizer.build(["hello world"])
    assert tok.itos[0] == BLANK_TOKEN
    assert tok.blank_id == 0


def test_build_is_deterministic():
    a = CharTokenizer.build(["cba", "abc"])
    b = CharTokenizer.build(["abc", "cba"])
    assert a.itos == b.itos


def test_encode_decode_roundtrip():
    tok = CharTokenizer.build(["patient denies chest pain"])
    text = "patient denies chest pain"
    assert tok.decode(tok.encode(text)) == text


def test_unknown_char_maps_to_unk_and_is_dropped_on_decode():
    tok = CharTokenizer.build(["abc"])  # no 'z'
    ids = tok.encode("azb")
    # 'z' -> unk id, which is filtered on decode
    assert tok.decode(ids) == "ab"


def test_save_and_load(tmp_path):
    tok = CharTokenizer.build(["metformin 500 mg"])
    path = tmp_path / "tok.json"
    tok.save(path)
    reloaded = CharTokenizer.load(path)
    assert reloaded.itos == tok.itos
    assert reloaded.vocab_size == tok.vocab_size
