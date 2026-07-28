import math

import pytest
import torch

from medasr.confidence import (
    aggregate_words,
    frame_entropy,
    score_utterance,
    token_confidences,
)
from medasr.tokenizer import CharTokenizer


def tok_for(text="cat hat"):
    return CharTokenizer.build([text])


def peaky(tokenizer, frame_tokens, sharpness=20.0):
    T, V = len(frame_tokens), tokenizer.vocab_size
    logits = torch.full((T, V), -sharpness)
    for t, tid in enumerate(frame_tokens):
        logits[t, tid] = sharpness
    return torch.log_softmax(logits, dim=-1)


def test_frame_entropy_bounds():
    V = 8
    uniform = torch.log(torch.full((V,), 1.0 / V))
    assert frame_entropy(uniform) == pytest.approx(math.log(V), abs=1e-5)

    onehot = torch.full((V,), -50.0)
    onehot[3] = 50.0
    assert frame_entropy(torch.log_softmax(onehot, dim=-1)) == pytest.approx(0.0, abs=1e-4)


def test_token_confidences_follow_ctc_collapse():
    tok = tok_for()
    c, a, t = (tok.stoi[x] for x in "cat")
    b = tok.blank_id
    lp = peaky(tok, [c, c, b, a, b, t, t])
    tokens = token_confidences(lp, tok)
    assert "".join(x.text for x in tokens) == "cat"
    assert [x.frame for x in tokens] == [0, 3, 5]
    assert all(x.probability > 0.9 for x in tokens)


def test_token_confidences_rejects_batched_input():
    tok = tok_for()
    with pytest.raises(ValueError):
        token_confidences(torch.zeros(1, 5, tok.vocab_size), tok)


def test_confident_input_scores_high():
    tok = tok_for()
    c, a, t = (tok.stoi[x] for x in "cat")
    report = score_utterance(peaky(tok, [c, a, t]), tok)
    assert report.text == "cat"
    assert report.utterance_confidence > 0.9
    assert not report.needs_review


def test_uncertain_input_is_flagged():
    tok = tok_for()
    torch.manual_seed(0)
    # Nearly uniform -> low probability and high entropy.
    lp = torch.log_softmax(torch.randn(12, tok.vocab_size) * 0.01, dim=-1)
    report = score_utterance(lp, tok)
    assert report.utterance_confidence < 0.5
    assert report.needs_review
    assert len(report.low_confidence_words) > 0


def test_word_confidence_uses_weakest_token():
    tok = tok_for()
    c, a, t = (tok.stoi[x] for x in "cat")
    # Make the middle character deliberately uncertain: 'a' only narrowly
    # beats a competing token, so its posterior is far below 1.
    V = tok.vocab_size
    logits = torch.full((3, V), -20.0)
    logits[0, c] = 20.0
    logits[1, a] = 1.0
    logits[1, t] = 0.9    # close rival -> 'a' wins but with low confidence
    logits[2, t] = 20.0
    lp = torch.log_softmax(logits, dim=-1)

    tokens = token_confidences(lp, tok)
    words = aggregate_words(tokens, threshold=0.6, entropy_threshold=99.0)
    assert len(words) == 1
    w = words[0]
    # Weakest-link: word prob equals the minimum token prob, below the mean.
    assert w.probability == pytest.approx(min(t_.probability for t_ in tokens))
    assert w.probability < w.mean_probability
    assert w.low_confidence


def test_words_split_on_space():
    tok = tok_for()
    ids = [tok.stoi[x] for x in "cat hat"]
    report = score_utterance(peaky(tok, ids), tok)
    assert [w.text for w in report.words] == ["cat", "hat"]


def test_empty_and_all_blank_input():
    tok = tok_for()
    blanks = peaky(tok, [tok.blank_id] * 5)
    report = score_utterance(blanks, tok)
    assert report.text == ""
    assert report.words == []
    assert report.utterance_confidence == 0.0
    assert not report.needs_review


def test_high_entropy_alone_triggers_review():
    tok = tok_for()
    c = tok.stoi["c"]
    V = tok.vocab_size
    # Top token is probable, but mass is spread widely behind it.
    logits = torch.full((1, V), 1.0)
    logits[0, c] = 3.0
    lp = torch.log_softmax(logits, dim=-1)
    tokens = token_confidences(lp, tok)
    words = aggregate_words(tokens, threshold=0.0, entropy_threshold=0.5)
    assert words and words[0].low_confidence
