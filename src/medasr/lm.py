"""A character n-gram language model for shallow fusion.

An acoustic model hears sounds; it has no idea that "patient denies chest
pain" is a thousand times more likely than "patient denies chest pane". A
language model supplies exactly that prior, and in clinical dictation the
prior is unusually strong — notes are formulaic, drug names recur, and the
phrasing is house style.

**Shallow fusion** combines the two at decode time by scoring each beam with

    score = log P_acoustic  +  alpha * log P_lm  +  beta * |tokens|

``alpha`` weights the LM. ``beta`` is an insertion bonus that counteracts the
LM's structural bias toward *shorter* strings (every extra token multiplies in
another probability < 1, so without ``beta`` the beam quietly prefers dropping
words — which in a medical note might be the word "no").

The model is a Katz-style back-off n-gram over **characters**, which pairs
naturally with the character CTC vocabulary and never encounters an unknown
word — important when the next token is a drug name the LM has never seen.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

BOS = "\x02"  # sentinel for "start of sentence" context


class CharNGramLM:
    """Back-off character n-gram LM with add-k smoothing.

    Deliberately dependency-free and small enough to ship next to a model
    checkpoint. For large corpora prefer KenLM; the :meth:`log_prob` interface
    is the same, so the beam search does not care which you use.
    """

    def __init__(self, order: int = 5, alpha_smoothing: float = 0.1):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.k = alpha_smoothing
        # counts[context][next_char] -> int
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.context_totals: Dict[str, int] = defaultdict(int)
        self.vocab: set[str] = set()

    # -- training ------------------------------------------------------------
    def train(self, texts: Iterable[str]) -> "CharNGramLM":
        """Accumulate counts for every order from 1..``order``."""
        for text in texts:
            padded = BOS * (self.order - 1) + text
            self.vocab.update(text)
            for i in range(self.order - 1, len(padded)):
                nxt = padded[i]
                # Every back-off order shares the same counting pass.
                for n in range(self.order):
                    ctx = padded[i - n : i]
                    self.counts[ctx][nxt] += 1
                    self.context_totals[ctx] += 1
        return self

    # -- scoring -------------------------------------------------------------
    def log_prob(self, context: str, char: str) -> float:
        """log P(char | context), backing off to shorter contexts."""
        v = max(1, len(self.vocab))
        # Try longest context first, shortening until we have evidence.
        ctx = context[-(self.order - 1) :] if self.order > 1 else ""
        while True:
            total = self.context_totals.get(ctx, 0)
            if total > 0:
                count = self.counts.get(ctx, {}).get(char, 0)
                return math.log((count + self.k) / (total + self.k * v))
            if not ctx:
                return math.log(1.0 / v)  # uniform fallback
            ctx = ctx[1:]

    def score(self, text: str) -> float:
        """Total log-probability of a string."""
        padded = BOS * (self.order - 1) + text
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            total += self.log_prob(padded[i - (self.order - 1) : i], padded[i])
        return total

    def perplexity(self, text: str) -> float:
        if not text:
            return float("inf")
        return math.exp(-self.score(text) / len(text))

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "order": self.order,
            "k": self.k,
            "vocab": sorted(self.vocab),
            "counts": {ctx: dict(nxt) for ctx, nxt in self.counts.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharNGramLM":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        lm = cls(order=payload["order"], alpha_smoothing=payload["k"])
        lm.vocab = set(payload["vocab"])
        for ctx, nxt in payload["counts"].items():
            for ch, c in nxt.items():
                lm.counts[ctx][ch] = c
                lm.context_totals[ctx] += c
        return lm
