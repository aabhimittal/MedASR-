"""PHI (Protected Health Information) detection and redaction.

Clinical dictation is dense with identifiers: medical record numbers, dates of
birth, phone numbers, addresses. The moment a transcript leaves the recognition
service — into a log line, an analytics pipeline, a model-improvement corpus —
those identifiers become a HIPAA problem. The cheapest place to solve it is
right where the text is produced.

This is a **deterministic, rule-based** de-identifier covering the structured
HIPAA Safe Harbor identifiers, which are the ones regular expressions actually
handle well: MRNs, SSNs, phone/fax, email, dates, ages over 89, ZIP codes, and
URLs. It is intentionally *not* a named-entity model.

**Read this before deploying it.** Rule-based de-identification has a real,
well-documented recall ceiling — it will not reliably catch free-text patient
or relative names, unusual institution names, or identifiers written in words
("record number three four seven"). Safe Harbor compliance requires removing
*all* eighteen identifier classes. Treat this module as a strong first pass
and defence-in-depth, not as a certification of de-identification. Anything
leaving a covered environment needs either expert determination or a validated
NER-based system layered on top.

Because recall is imperfect, the API is built around *explicitness*: you always
get back a :class:`RedactionReport` telling you exactly what was found and
replaced, so a caller can log counts (never contents) and route uncertain text
to human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


@dataclass(frozen=True)
class PHIMatch:
    """One detected identifier. ``text`` is the raw match — handle with care."""

    kind: str
    text: str
    start: int
    end: int


@dataclass
class RedactionReport:
    redacted_text: str
    matches: List[PHIMatch] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        """Per-kind counts — safe to log, unlike the matches themselves."""
        out: Dict[str, int] = {}
        for m in self.matches:
            out[m.kind] = out.get(m.kind, 0) + 1
        return out

    @property
    def found_phi(self) -> bool:
        return bool(self.matches)


# Order matters: the first pattern to claim a span wins, so the most specific
# identifiers are listed before the more permissive numeric ones.
_PATTERNS: Tuple[Tuple[str, Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("URL", re.compile(r"\bhttps?://\S+\b", re.I)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "MRN",
        re.compile(
            r"\b(?:mrn|medical\s+record(?:\s+(?:number|no\.?|#))?|record\s+number)"
            r"\s*[:#]?\s*([A-Z]?\d{5,10})\b",
            re.I,
        ),
    ),
    (
        "PHONE",
        # Lookaround rather than \b: a leading "(" is a non-word character, so
        # "\b" never fires after a space and the very common "(555) 123-4567"
        # format would slip through unredacted.
        re.compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
        ),
    ),
    (
        "DATE",
        re.compile(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
            r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
            re.I,
        ),
    ),
    # HIPAA: ages of 90+ are identifying and must be generalised.
    ("AGE_OVER_89", re.compile(r"\b(?:9\d|1\d{2})\s*(?:-|\s)?year[s]?[\s-]old\b", re.I)),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
)

_PLACEHOLDER = {
    "EMAIL": "[EMAIL]",
    "URL": "[URL]",
    "SSN": "[SSN]",
    "MRN": "[MRN]",
    "PHONE": "[PHONE]",
    "DATE": "[DATE]",
    "AGE_OVER_89": "[AGE>89]",
    "ZIP": "[ZIP]",
}


def find_phi(text: str) -> List[PHIMatch]:
    """Locate identifiers, resolving overlaps in favour of the earlier pattern."""
    claimed: List[Tuple[int, int]] = []
    matches: List[PHIMatch] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < ce and e > cs for cs, ce in claimed)

    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            # For cue-based patterns (MRN) redact only the identifier group.
            if m.groups():
                s, e = m.span(1)
            else:
                s, e = m.span()
            if overlaps(s, e):
                continue
            claimed.append((s, e))
            matches.append(PHIMatch(kind=kind, text=text[s:e], start=s, end=e))

    matches.sort(key=lambda m: m.start)
    return matches


def redact(text: str, placeholder: Dict[str, str] | None = None) -> RedactionReport:
    """Replace detected identifiers with typed placeholders.

    Replacement runs right-to-left so earlier offsets stay valid.
    """
    ph = {**_PLACEHOLDER, **(placeholder or {})}
    matches = find_phi(text)

    out = text
    for m in sorted(matches, key=lambda m: m.start, reverse=True):
        out = out[: m.start] + ph.get(m.kind, "[REDACTED]") + out[m.end :]

    return RedactionReport(redacted_text=out, matches=matches)


def contains_phi(text: str) -> bool:
    """Fast boolean check — useful as a guard before logging a transcript."""
    return bool(find_phi(text))
