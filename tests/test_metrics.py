from medasr.metrics import cer, char_errors, wer, word_errors


def test_perfect_match_is_zero():
    assert wer(["hello world"], ["hello world"]) == 0.0
    assert cer(["hello"], ["hello"]) == 0.0


def test_single_substitution():
    er = word_errors("the cat sat", "the dog sat")
    assert er.errors == 1 and er.total == 3
    assert abs(er.rate - 1 / 3) < 1e-9


def test_insertion_and_deletion():
    # ref 2 words, hyp 3 -> one insertion
    assert abs(wer(["take two pills"], ["take two more pills"]) - 1 / 3) < 1e-9
    # deletion
    assert abs(wer(["take two pills"], ["take pills"]) - 1 / 3) < 1e-9


def test_cer_catches_drug_name_slip():
    # One-character error that WER counts as a whole word.
    er = char_errors("hydralazine", "hydroxyzine")
    assert 0 < er.rate < 1
    assert wer(["give hydralazine"], ["give hydroxyzine"]) == 0.5


def test_corpus_micro_average():
    refs = ["a b c", "d e"]
    hyps = ["a x c", "d e"]
    # 1 error over 5 words
    assert abs(wer(refs, hyps) - 1 / 5) < 1e-9
