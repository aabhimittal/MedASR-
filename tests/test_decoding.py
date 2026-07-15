import torch

from medasr.decoding import _collapse, beam_search_decode, greedy_decode
from medasr.tokenizer import CharTokenizer


def make_tokenizer():
    # vocab: 0=<blank>, 1=<unk>, then sorted chars of "helo"
    return CharTokenizer.build(["hello"])


def test_collapse_rule():
    tok = make_tokenizer()
    # ids for h,e,l,o
    h, e, l, o = (tok.stoi[c] for c in "helo")
    blank = tok.blank_id
    # h h <b> e l l <b> l o  -> h e l l o
    path = [h, h, blank, e, l, l, blank, l, o]
    assert _collapse(path, tok) == "hello"


def test_greedy_decode_matches_argmax_path():
    tok = make_tokenizer()
    V = tok.vocab_size
    # Build log-probs that clearly favour "hi"-like sequence.
    T = 6
    logits = torch.full((1, T, V), -10.0)
    h = tok.stoi["h"]
    e = tok.stoi["e"]
    seq = [h, h, tok.blank_id, e, e, tok.blank_id]
    for t, s in enumerate(seq):
        logits[0, t, s] = 10.0
    log_probs = torch.log_softmax(logits, dim=-1)
    out = greedy_decode(log_probs, torch.tensor([T]), tok)
    assert out[0] == "he"


def test_beam_and_greedy_agree_on_confident_input():
    tok = make_tokenizer()
    V = tok.vocab_size
    T = 5
    logits = torch.full((1, T, V), -10.0)
    l = tok.stoi["l"]
    o = tok.stoi["o"]
    seq = [l, tok.blank_id, o, o, tok.blank_id]
    for t, s in enumerate(seq):
        logits[0, t, s] = 10.0
    log_probs = torch.log_softmax(logits, dim=-1)
    greedy = greedy_decode(log_probs, torch.tensor([T]), tok)
    beam = beam_search_decode(log_probs, torch.tensor([T]), tok, beam_size=4)
    assert greedy == beam == ["lo"]


def test_length_masking_ignores_padded_frames():
    tok = make_tokenizer()
    V = tok.vocab_size
    log_probs = torch.log_softmax(torch.randn(1, 10, V), dim=-1)
    # Only first 3 frames are valid.
    short = greedy_decode(log_probs, torch.tensor([3]), tok)
    full = greedy_decode(log_probs, torch.tensor([10]), tok)
    assert len(short[0]) <= len(full[0])
