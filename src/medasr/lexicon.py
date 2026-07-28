"""Medical lexicon: drug-name correction and clinical safety checks.

A character-level CTC model spells phonetically. Given unfamiliar audio it will
happily emit ``metfarmin`` or ``lisinapril`` — close enough that a reader
understands, wrong enough that downstream code matching against a formulary
fails. A lexicon pass fixes those near-misses.

But the *interesting* problem is the opposite one. Some drug pairs are both
real words and acoustically adjacent:

    hydralazine (antihypertensive)  vs  hydroxyzine (antihistamine)
    clonidine   (antihypertensive)  vs  klonopin    (benzodiazepine)

Here a spell-corrector is actively dangerous: both spellings are valid, so
"correcting" one to the other silently changes the prescription. These are
tracked separately in :data:`CONFUSABLE_PAIRS` and are **never auto-corrected**
— they are flagged for human confirmation. This is the ISMP "confused drug
names" concept encoded as data.

The dosage checks catch the other classic dictation hazard: a misheard decimal
point. ``1.5 mg`` and ``15 mg`` are one character apart and an order of
magnitude apart in effect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from medasr.metrics import _levenshtein

# A small starter formulary. Real deployments load a full RxNorm/formulary
# export; the structure is identical.
DEFAULT_DRUGS: Tuple[str, ...] = (
    "amoxicillin", "atorvastatin", "amlodipine", "albuterol", "aspirin",
    "ceftriaxone", "ciprofloxacin", "clonidine", "clopidogrel",
    "furosemide", "gabapentin", "hydralazine", "hydrochlorothiazide",
    "hydroxyzine", "ibuprofen", "insulin", "levothyroxine", "lisinopril",
    "losartan", "metformin", "metoprolol", "morphine", "naloxone",
    "omeprazole", "ondansetron", "oxycodone", "penicillin", "prednisone",
    "sertraline", "simvastatin", "tramadol", "warfarin",
)

# Pairs that sound alike but are clinically distinct. Never auto-correct
# between them; always surface for confirmation.
CONFUSABLE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("hydralazine", "hydroxyzine"),
    ("clonidine", "klonopin"),
    ("metformin", "metronidazole"),
    ("prednisone", "prednisolone"),
    ("morphine", "hydromorphone"),
    ("oxycodone", "oxycontin"),
    ("losartan", "valsartan"),
    ("simvastatin", "atorvastatin"),
)

# Units we understand for dose sanity checks.
_DOSE_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|units?|iu)\b", re.I
)

# Plausible adult single-dose ceilings, in the unit given. Exceeding these is
# not necessarily wrong, but it is worth a human glance.
_DOSE_CEILING: Dict[str, float] = {
    "mg": 4000.0,
    "mcg": 1000.0,
    "g": 10.0,
    "ml": 1000.0,
    "unit": 500.0,
    "units": 500.0,
    "iu": 500.0,
}


@dataclass
class Correction:
    """A single word the lexicon changed."""

    original: str
    corrected: str
    distance: int


@dataclass
class SafetyFlag:
    """Something a human should confirm before this text is trusted."""

    kind: str          # "confusable" | "dose_range" | "ambiguous_decimal"
    text: str
    message: str


@dataclass
class LexiconResult:
    text: str
    corrections: List[Correction] = field(default_factory=list)
    flags: List[SafetyFlag] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


def _build_confusable_index(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Set[str]]:
    index: Dict[str, Set[str]] = {}
    for a, b in pairs:
        index.setdefault(a, set()).add(b)
        index.setdefault(b, set()).add(a)
    return index


class MedicalLexicon:
    """Fuzzy-corrects drug names and raises clinical safety flags."""

    def __init__(
        self,
        terms: Iterable[str] = DEFAULT_DRUGS,
        confusable_pairs: Sequence[Tuple[str, str]] = CONFUSABLE_PAIRS,
        max_distance: int = 2,
        min_length: int = 5,
    ):
        self.terms: Set[str] = {t.lower() for t in terms}
        self.confusables = _build_confusable_index(confusable_pairs)
        # Every member of a confusable pair is a legitimate word.
        self.terms.update(self.confusables)
        self.max_distance = max_distance
        self.min_length = min_length

    # -- correction ----------------------------------------------------------
    def suggest(self, word: str) -> Optional[Tuple[str, int]]:
        """Nearest in-lexicon term within ``max_distance``, or None.

        Returns None when the match is **ambiguous** (two different terms tie
        at the same distance). Guessing between two drug names is precisely the
        error mode we must not introduce.
        """
        w = word.lower()
        if len(w) < self.min_length or w in self.terms:
            return None

        # Scale tolerance with length: short words are too easy to mangle.
        budget = 1 if len(w) < 8 else self.max_distance
        best: List[Tuple[str, int]] = []
        best_d = budget + 1
        for term in self.terms:
            # Cheap length gate before the O(n*m) edit distance.
            if abs(len(term) - len(w)) > budget:
                continue
            d = _levenshtein(w, term)
            if d < best_d:
                best_d, best = d, [(term, d)]
            elif d == best_d:
                best.append((term, d))

        if best_d > budget or not best:
            return None
        if len({t for t, _ in best}) > 1:
            return None  # ambiguous — refuse to guess
        return best[0]

    # -- safety --------------------------------------------------------------
    def confusable_for(self, word: str) -> Set[str]:
        return self.confusables.get(word.lower(), set())

    def check_doses(self, text: str) -> List[SafetyFlag]:
        """Flag implausible magnitudes and ambiguous decimals."""
        flags: List[SafetyFlag] = []
        for m in _DOSE_RE.finditer(text):
            value = float(m.group("value"))
            unit = m.group("unit").lower()
            ceiling = _DOSE_CEILING.get(unit)
            if ceiling is not None and value > ceiling:
                flags.append(
                    SafetyFlag(
                        kind="dose_range",
                        text=m.group(0),
                        message=(
                            f"{m.group(0)} exceeds the typical single-dose ceiling "
                            f"of {ceiling:g} {unit}; confirm before use."
                        ),
                    )
                )
            # A leading decimal ("0.5 mg" written as ".5 mg") or a sub-unit
            # dose is where a dropped/added point does the most damage.
            if "." in m.group("value") and value < 1.0:
                flags.append(
                    SafetyFlag(
                        kind="ambiguous_decimal",
                        text=m.group(0),
                        message=(
                            f"Sub-unit dose {m.group(0)}: confirm the decimal point "
                            "(a dropped point changes the dose 10-fold)."
                        ),
                    )
                )
        return flags

    # -- pipeline ------------------------------------------------------------
    def apply(self, text: str, correct: bool = True) -> LexiconResult:
        """Correct near-miss drug names and collect safety flags."""
        result = LexiconResult(text=text)
        out_tokens: List[str] = []

        for token in re.split(r"(\W+)", text):
            if not token or not token.isalpha():
                out_tokens.append(token)
                continue

            lower = token.lower()
            if lower in self.confusables:
                others = ", ".join(sorted(self.confusables[lower]))
                result.flags.append(
                    SafetyFlag(
                        kind="confusable",
                        text=token,
                        message=(
                            f"'{token}' is easily confused with: {others}. "
                            "Confirm the intended drug."
                        ),
                    )
                )
                out_tokens.append(token)
                continue

            suggestion = self.suggest(token) if correct else None
            if suggestion is None:
                out_tokens.append(token)
                continue

            term, dist = suggestion
            # Preserve the original capitalisation style.
            fixed = term.capitalize() if token[0].isupper() else term
            out_tokens.append(fixed)
            result.corrections.append(Correction(token, fixed, dist))

        result.text = "".join(out_tokens)
        result.flags.extend(self.check_doses(result.text))
        return result
