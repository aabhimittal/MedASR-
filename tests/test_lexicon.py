from medasr.lexicon import CONFUSABLE_PAIRS, MedicalLexicon


def lex():
    return MedicalLexicon()


# -- correction --------------------------------------------------------------
def test_corrects_near_miss_drug_name():
    r = lex().apply("start metfarmin today")
    assert "metformin" in r.text
    assert [(c.original, c.corrected) for c in r.corrections] == [("metfarmin", "metformin")]


def test_known_term_untouched():
    r = lex().apply("start metformin today")
    assert r.text == "start metformin today"
    assert r.corrections == []


def test_short_words_are_never_corrected():
    """Common English must survive: 'the', 'and' must not become drug names."""
    text = "the patient and his son are here for a new dose"
    r = lex().apply(text)
    assert r.text == text
    assert r.corrections == []


def test_distant_word_not_corrected():
    r = lex().apply("xylophone")
    assert r.text == "xylophone"
    assert r.corrections == []


def test_capitalisation_preserved():
    r = lex().apply("Metfarmin daily")
    assert r.text.startswith("Metformin")


def test_correction_disabled():
    r = lex().apply("metfarmin", correct=False)
    assert r.text == "metfarmin"
    assert r.corrections == []


def test_punctuation_and_spacing_preserved():
    r = lex().apply("give metformin, 500 mg.")
    assert r.text == "give metformin, 500 mg."


# -- confusable safety -------------------------------------------------------
def test_confusable_drug_is_flagged_not_corrected():
    """The critical safety property: never silently swap look-alike drugs."""
    r = lex().apply("start hydralazine 25 mg")
    assert "hydralazine" in r.text          # unchanged
    assert not r.corrections
    kinds = [f.kind for f in r.flags]
    assert "confusable" in kinds
    msg = next(f.message for f in r.flags if f.kind == "confusable")
    assert "hydroxyzine" in msg


def test_both_members_of_a_pair_are_flagged():
    l = lex()
    for a, b in CONFUSABLE_PAIRS:
        assert l.confusable_for(a), f"{a} should have a confusable partner"
        assert l.confusable_for(b), f"{b} should have a confusable partner"
        assert b in l.confusable_for(a)
        assert a in l.confusable_for(b)


def test_confusable_never_auto_corrected_into_partner():
    l = lex()
    for a, b in CONFUSABLE_PAIRS:
        assert l.suggest(a) is None
        assert l.suggest(b) is None


def test_ambiguous_suggestion_is_refused():
    """Equidistant candidates must yield no correction rather than a coin flip."""
    l = MedicalLexicon(terms=["warfarin", "warfarim"], max_distance=2, min_length=3)
    assert l.suggest("warfarix") is None


# -- dose safety -------------------------------------------------------------
def test_excessive_dose_flagged():
    r = lex().apply("give 9000 mg")
    assert any(f.kind == "dose_range" for f in r.flags)


def test_normal_dose_not_flagged():
    r = lex().apply("give 500 mg")
    assert not any(f.kind == "dose_range" for f in r.flags)


def test_sub_unit_decimal_flagged():
    r = lex().apply("give 0.5 mg")
    flags = [f for f in r.flags if f.kind == "ambiguous_decimal"]
    assert flags and "decimal" in flags[0].message


def test_whole_number_dose_not_decimal_flagged():
    r = lex().apply("give 15 mg")
    assert not any(f.kind == "ambiguous_decimal" for f in r.flags)


def test_multiple_units_supported():
    for dose in ["250 mcg", "2 g", "10 ml", "20 units"]:
        r = lex().apply(f"give {dose}")
        assert not any(f.kind == "dose_range" for f in r.flags), dose


def test_needs_review_property():
    assert lex().apply("give hydralazine").needs_review
    assert not lex().apply("give amoxicillin 500 mg").needs_review


def test_empty_input():
    r = lex().apply("")
    assert r.text == "" and not r.flags and not r.corrections
