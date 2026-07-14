from medasr.text import (
    canonicalize_units,
    expand_spoken_punctuation,
    normalize,
    strip_accents,
)


def test_strip_accents():
    assert strip_accents("naïve café") == "naive cafe"


def test_expand_spoken_punctuation():
    assert expand_spoken_punctuation("no acute distress period") == "no acute distress ."
    # A real word containing a command substring is untouched.
    assert "commanded" in expand_spoken_punctuation("he commanded rest")


def test_canonicalize_units():
    assert canonicalize_units("500 milligrams") == "500 mg"
    assert canonicalize_units("2 MICROGRAMS") == "2 mcg"


def test_normalize_pipeline():
    out = normalize("Start Metformin 500 milligrams period")
    assert out == "start metformin 500 mg."


def test_normalize_collapses_whitespace():
    assert normalize("too    many   spaces") == "too many spaces"


def test_normalize_is_idempotent():
    once = normalize("Blood pressure 130 over 85 comma stable")
    assert normalize(once) == once
