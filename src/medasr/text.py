"""Medical text normalisation.

ASR models learn a mapping from audio to a *canonical* written form. If the
same spoken content appears as "5 mg", "5mg" and "five milligrams" in the
training transcripts, the model is forced to waste capacity modelling
orthographic noise instead of acoustics. Normalisation collapses these
variants to one form so the labels are consistent.

For **clinical dictation** there are domain-specific concerns:

* Dosages and units ("mg", "mcg", "ml", "b.i.d.", "p.r.n.").
* Abbreviations that are spoken in full ("BP" -> "blood pressure" is *not*
  applied here — we keep the written clinical shorthand, but we *do* normalise
  punctuation/spacing around it).
* Spoken punctuation commands ("period", "comma", "new paragraph") that a
  dictation system must turn into symbols.

This module keeps normalisation deterministic and reversible-in-spirit so the
same function is applied to training transcripts and to model output before
scoring.
"""

from __future__ import annotations

import re
import unicodedata

# Spoken dictation punctuation commands -> written symbol.
# Clinicians literally say "period" / "comma" while dictating.
_SPOKEN_PUNCTUATION = {
    "full stop": ".",
    "period": ".",
    "comma": ",",
    "colon": ":",
    "semicolon": ";",
    "question mark": "?",
    "exclamation mark": "!",
    "open paren": "(",
    "close paren": ")",
    "new paragraph": "\n",
    "new line": "\n",
}

# Common clinical unit spellings -> canonical short form.
_UNIT_CANON = {
    "milligrams": "mg",
    "milligram": "mg",
    "micrograms": "mcg",
    "microgram": "mcg",
    "millilitres": "ml",
    "milliliters": "ml",
    "milliliter": "ml",
    "grams": "g",
    "gram": "g",
    "kilograms": "kg",
    "kilogram": "kg",
}

_MULTISPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,:;?!])")


def strip_accents(text: str) -> str:
    """Fold accented characters to ASCII (e.g. 'naïve' -> 'naive')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def expand_spoken_punctuation(text: str) -> str:
    """Turn dictated punctuation words into symbols.

    Only whole-word, case-insensitive matches are replaced so that a real word
    like "commanded" is left untouched.
    """
    def repl(match: re.Match) -> str:
        return _SPOKEN_PUNCTUATION[match.group(0).lower()]

    # Longest phrases first so "new paragraph" beats a hypothetical "new".
    keys = sorted(_SPOKEN_PUNCTUATION, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.I)
    return pattern.sub(repl, text)


def canonicalize_units(text: str) -> str:
    """Map spelled-out clinical units to their canonical abbreviation."""
    def repl(match: re.Match) -> str:
        return _UNIT_CANON[match.group(0).lower()]

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in _UNIT_CANON) + r")\b", re.I
    )
    return pattern.sub(repl, text)


def normalize(
    text: str,
    *,
    lowercase: bool = True,
    expand_punctuation: bool = True,
    canon_units: bool = True,
) -> str:
    """Full normalisation pipeline used for training labels and scoring.

    The order matters: expand spoken commands first (they may introduce
    punctuation), then units, then collapse whitespace/spacing.
    """
    text = strip_accents(text)
    if expand_punctuation:
        text = expand_spoken_punctuation(text)
    if canon_units:
        text = canonicalize_units(text)
    if lowercase:
        text = text.lower()
    text = _MULTISPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()
