"""Evaluation metrics: Word Error Rate (WER) and Character Error Rate (CER).

ASR quality is measured by how much the hypothesis must be edited to match the
reference. Both metrics are the Levenshtein (edit) distance normalised by the
reference length::

    WER = (substitutions + insertions + deletions) / number_of_reference_words

WER operates on whitespace-split words; CER on characters. **CER matters a lot
for medical ASR**: a one-character slip in a drug name or dosage ("hydralazine"
vs "hydroxyzine", "15 mg" vs "1.5 mg") is a clinically critical error that WER
counts the same as any other word substitution, whereas CER exposes how close
(or far) the spelling really was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


def _levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Classic O(n*m) edit distance with a rolling row."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = curr
    return prev[m]


@dataclass
class ErrorRate:
    errors: int
    total: int

    @property
    def rate(self) -> float:
        return self.errors / self.total if self.total else 0.0

    def __add__(self, other: "ErrorRate") -> "ErrorRate":
        return ErrorRate(self.errors + other.errors, self.total + other.total)


def word_errors(reference: str, hypothesis: str) -> ErrorRate:
    ref, hyp = reference.split(), hypothesis.split()
    return ErrorRate(_levenshtein(ref, hyp), len(ref))


def char_errors(reference: str, hypothesis: str) -> ErrorRate:
    return ErrorRate(_levenshtein(list(reference), list(hypothesis)), len(reference))


def wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level WER (micro-average over all reference words)."""
    total = ErrorRate(0, 0)
    for ref, hyp in zip(references, hypotheses):
        total = total + word_errors(ref, hyp)
    return total.rate


def cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level CER (micro-average over all reference characters)."""
    total = ErrorRate(0, 0)
    for ref, hyp in zip(references, hypotheses):
        total = total + char_errors(ref, hyp)
    return total.rate
