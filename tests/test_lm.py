import math

import pytest
import torch

from medasr.decoding import beam_search_decode, greedy_decode
from medasr.lm import CharNGramLM
from medasr.tokenizer import CharTokenizer

CORPUS = [
    "patient denies chest pain",
    "patient denies shortness of breath",
    "no known drug allergies",
    "start metformin 500 mg twice daily",
]


def test_rejects_bad_order():
    with pytest.raises(ValueError):
        CharNGramLM(order=0)


def test_probabilities_are_valid_and_normalised():
    lm = CharNGramLM(order=3).train(CORPUS)
    total = sum(math.exp(lm.log_prob("pa", c)) for c in lm.vocab)
    assert total <= 1.0 + 1e-6
    for c in lm.vocab:
        assert lm.log_prob("pa", c) <= 0.0


def test_in_domain_text_scores_better():
    lm = CharNGramLM(order=4).train(CORPUS)
    assert lm.perplexity("patient denies chest pain") < lm.perplexity("qqqq zzzz xxxx")


def test_unseen_context_backs_off_without_crashing():
    lm = CharNGramLM(order=5).train(CORPUS)
    lp = lm.log_prob("zzzzq", "a")
    assert math.isfinite(lp) and lp < 0.0


def test_unseen_char_gets_finite_probability():
    """Never return -inf: one impossible char must not kill an entire beam."""
    lm = CharNGramLM(order=3).train(CORPUS)
    assert math.isfinite(lm.log_prob("pa", "ñ"))


def test_empty_text_perplexity_is_inf():
    lm = CharNGramLM(order=3).train(CORPUS)
    assert lm.perplexity("") == float("inf")


def test_unigram_order_one_works():
    lm = CharNGramLM(order=1).train(CORPUS)
    assert math.isfinite(lm.score("patient"))


def test_score_is_sum_of_log_probs():
    lm = CharNGramLM(order=2).train(CORPUS)
    assert lm.score("ab") == pytest.approx(lm.score("a") + lm.log_prob("a", "b"), abs=1e-9)


def test_save_and_load_roundtrip(tmp_path):
    lm = CharNGramLM(order=3).train(CORPUS)
    path = tmp_path / "lm.json"
    lm.save(path)
    loaded = CharNGramLM.load(path)
    assert loaded.order == lm.order
    assert loaded.vocab == lm.vocab
    text = "patient denies"
    assert loaded.score(text) == pytest.approx(lm.score(text), abs=1e-9)


# -- shallow fusion in the decoder ------------------------------------------
def test_fusion_runs_and_returns_text():
    tok = CharTokenizer.build(CORPUS)
    lm = CharNGramLM(order=4).train(CORPUS)
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(1, 10, tok.vocab_size), dim=-1)
    out = beam_search_decode(lp, torch.tensor([10]), tok, 4, lm, 0.5, 0.5)
    assert isinstance(out[0], str)


def test_zero_lm_weight_matches_plain_beam():
    """Fusion with weight 0 must be exactly the unfused search."""
    tok = CharTokenizer.build(CORPUS)
    lm = CharNGramLM(order=3).train(CORPUS)
    torch.manual_seed(1)
    lp = torch.log_softmax(torch.randn(1, 12, tok.vocab_size), dim=-1)
    lengths = torch.tensor([12])
    assert beam_search_decode(lp, lengths, tok, 5) == beam_search_decode(
        lp, lengths, tok, 5, lm, 0.0, 0.0
    )


def test_fusion_leaves_confident_acoustics_alone():
    """A clear acoustic signal must not be overridden by the LM."""
    tok = CharTokenizer.build(CORPUS)
    lm = CharNGramLM(order=4).train(CORPUS)
    p, a, i = tok.stoi["p"], tok.stoi["a"], tok.stoi["i"]
    V, seq = tok.vocab_size, [p, tok.blank_id, a, tok.blank_id, i]
    logits = torch.full((1, len(seq), V), -30.0)
    for t, s in enumerate(seq):
        logits[0, t, s] = 30.0
    lp = torch.log_softmax(logits, dim=-1)
    lengths = torch.tensor([len(seq)])
    assert greedy_decode(lp, lengths, tok) == ["pai"]
    assert beam_search_decode(lp, lengths, tok, 4, lm, 0.4, 0.3) == ["pai"]


def test_insertion_bonus_does_not_shorten_output():
    """A positive bonus must never produce a shorter hypothesis."""
    tok = CharTokenizer.build(CORPUS)
    lm = CharNGramLM(order=3).train(CORPUS)
    torch.manual_seed(2)
    lp = torch.log_softmax(torch.randn(1, 14, tok.vocab_size) * 2, dim=-1)
    lengths = torch.tensor([14])
    none = beam_search_decode(lp, lengths, tok, 6, lm, 0.6, 0.0)[0]
    bonus = beam_search_decode(lp, lengths, tok, 6, lm, 0.6, 1.0)[0]
    assert len(bonus) >= len(none)
