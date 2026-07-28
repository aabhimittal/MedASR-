import pytest
import torch

from medasr.alignment import (
    align_tokens,
    forced_align,
    frames_to_seconds,
    group_words,
)
from medasr.tokenizer import CharTokenizer


def tok_for(text="hello world cat"):
    return CharTokenizer.build([text])


def peaky_log_probs(tokenizer, frame_tokens):
    """Build near-deterministic log-probs emitting `frame_tokens` per frame."""
    T, V = len(frame_tokens), tokenizer.vocab_size
    logits = torch.full((T, V), -20.0)
    for t, tid in enumerate(frame_tokens):
        logits[t, tid] = 20.0
    return torch.log_softmax(logits, dim=-1)


def test_frames_to_seconds():
    # frame 25 at hop 160, sr 16k, subsample 4 -> 25*4*160/16000 = 1.0 s
    assert frames_to_seconds(25, 160, 16000, 4) == pytest.approx(1.0)
    assert frames_to_seconds(0, 160, 16000, 4) == 0.0


def test_forced_align_path_length_matches_frames():
    tok = tok_for()
    ids = tok.encode("cat")
    lp = torch.log_softmax(torch.randn(20, tok.vocab_size), dim=-1)
    path = forced_align(lp, ids, tok.blank_id)
    assert len(path) == 20
    assert all(0 <= s < 2 * len(ids) + 1 for s in path)


def test_forced_align_empty_target():
    tok = tok_for()
    lp = torch.log_softmax(torch.randn(5, tok.vocab_size), dim=-1)
    assert forced_align(lp, [], tok.blank_id) == [0] * 5


def test_forced_align_recovers_known_alignment():
    tok = tok_for()
    c, a, t = (tok.stoi[x] for x in "cat")
    b = tok.blank_id
    # A clean emission: c c <b> a a <b> t t
    lp = peaky_log_probs(tok, [c, c, b, a, a, b, t, t])
    spans = align_tokens(lp, [c, a, t], tok)
    assert [s.text for s in spans] == ["c", "a", "t"]
    assert spans[0].start_frame == 0
    assert spans[1].start_frame == 3
    assert spans[2].start_frame == 6
    # Spans must be ordered and non-overlapping.
    for x, y in zip(spans, spans[1:]):
        assert x.end_frame <= y.start_frame


def test_forced_align_raises_when_audio_too_short():
    tok = tok_for()
    lp = torch.log_softmax(torch.randn(2, tok.vocab_size), dim=-1)
    with pytest.raises(ValueError, match="too short"):
        forced_align(lp, tok.encode("cat"), tok.blank_id)


def test_repeated_chars_need_an_extra_frame():
    """'ll' requires a blank between the two l's -> 3 frames minimum."""
    tok = CharTokenizer.build(["ll"])
    ids = tok.encode("ll")
    lp2 = torch.log_softmax(torch.randn(2, tok.vocab_size), dim=-1)
    with pytest.raises(ValueError):
        forced_align(lp2, ids, tok.blank_id)
    lp3 = torch.log_softmax(torch.randn(3, tok.vocab_size), dim=-1)
    assert len(forced_align(lp3, ids, tok.blank_id)) == 3


def test_timestamps_are_monotonic_and_bounded():
    tok = tok_for()
    ids = tok.encode("hello")
    lp = torch.log_softmax(torch.randn(40, tok.vocab_size), dim=-1)
    spans = align_tokens(lp, ids, tok, hop_length=160, sample_rate=16000)
    for s in spans:
        assert s.start_s <= s.end_s
    for x, y in zip(spans, spans[1:]):
        assert x.start_s <= y.start_s


def test_group_words_splits_on_space():
    tok = tok_for()
    ids = tok.encode("cat hat")
    lp = torch.log_softmax(torch.randn(30, tok.vocab_size), dim=-1)
    words = group_words(align_tokens(lp, ids, tok))
    assert [w.text for w in words] == ["cat", "hat"]
    assert all(0.0 <= w.score <= 1.0 for w in words)
    assert all(w.duration_s >= 0 for w in words)


def test_group_words_empty_input():
    assert group_words([]) == []
