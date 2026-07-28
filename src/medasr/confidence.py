"""Confidence estimation for recognised text.

A transcript without confidence is a liability. The model will always output
*something*, and a Conformer is perfectly capable of emitting "hydroxyzine"
with the same typography as a word it actually heard clearly. For clinical
dictation the useful question is not "what did the model say" but **"which
parts should a human look at before this enters the chart"**.

Three quantities, each answering a different question:

* **Token confidence** — the posterior at the frame that emitted the token.
* **Word confidence** — aggregated over a word's tokens. We aggregate with the
  **minimum**, not the mean: a word is only as trustworthy as its weakest
  character, because that is exactly where "15" becomes "1.5".
* **Utterance confidence** — the length-normalised mean, for routing whole
  clips to human review.

We also expose **entropy**, which catches a failure mode probability alone
misses: a frame can put 0.5 on the right token while spreading the rest over
ten plausible alternatives (genuinely uncertain) or over one (a clean
two-way toss-up). High entropy across a word is a reliable "the model was
guessing" signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch

from medasr.tokenizer import BaseTokenizer


@dataclass
class TokenConfidence:
    token_id: int
    text: str
    frame: int
    probability: float
    entropy: float


@dataclass
class WordConfidence:
    text: str
    probability: float      # min over the word's tokens (weakest link)
    mean_probability: float
    max_entropy: float
    low_confidence: bool

    @property
    def needs_review(self) -> bool:
        return self.low_confidence


@dataclass
class ConfidenceReport:
    text: str
    utterance_confidence: float
    words: List[WordConfidence]

    @property
    def low_confidence_words(self) -> List[WordConfidence]:
        return [w for w in self.words if w.low_confidence]

    @property
    def needs_review(self) -> bool:
        """True when any word should be checked by a human."""
        return bool(self.low_confidence_words)


def frame_entropy(log_probs_frame: torch.Tensor) -> float:
    """Shannon entropy (nats) of one frame's distribution."""
    p = log_probs_frame.exp()
    return float(-(p * log_probs_frame).sum())


def token_confidences(
    log_probs: torch.Tensor,
    tokenizer: BaseTokenizer,
) -> List[TokenConfidence]:
    """Greedy-decode one utterance and score each emitted token.

    Mirrors the CTC collapse rule so the returned tokens correspond exactly to
    the characters in the greedy transcript: a token is *emitted* at the first
    frame of a run that differs from the previous frame's argmax and is not
    the blank.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"Expected (T, V) log-probs, got shape {tuple(log_probs.shape)}")

    best = log_probs.argmax(dim=-1)
    out: List[TokenConfidence] = []
    prev = None
    for t in range(log_probs.size(0)):
        idx = int(best[t])
        if idx != prev and idx != tokenizer.blank_id:
            out.append(
                TokenConfidence(
                    token_id=idx,
                    text=tokenizer.decode([idx]),
                    frame=t,
                    probability=float(log_probs[t, idx].exp()),
                    entropy=frame_entropy(log_probs[t]),
                )
            )
        prev = idx
    return out


def aggregate_words(
    tokens: Sequence[TokenConfidence],
    threshold: float = 0.55,
    entropy_threshold: float = 1.5,
    separator: str = " ",
) -> List[WordConfidence]:
    """Group token confidences into words and flag the risky ones.

    A word is flagged when its weakest token falls below ``threshold`` *or*
    the model was diffuse (entropy above ``entropy_threshold``) anywhere in it.
    Either condition alone is enough — we would much rather over-flag a word
    than let a wrong dose through unchallenged.
    """
    words: List[WordConfidence] = []
    buf: List[TokenConfidence] = []

    def flush() -> None:
        if not buf:
            return
        text = "".join(t.text for t in buf)
        if text.strip():
            probs = [t.probability for t in buf]
            min_p = min(probs)
            max_h = max(t.entropy for t in buf)
            words.append(
                WordConfidence(
                    text=text,
                    probability=min_p,
                    mean_probability=sum(probs) / len(probs),
                    max_entropy=max_h,
                    low_confidence=(min_p < threshold or max_h > entropy_threshold),
                )
            )
        buf.clear()

    for tok in tokens:
        if tok.text == separator:
            flush()
        else:
            buf.append(tok)
    flush()
    return words


def score_utterance(
    log_probs: torch.Tensor,
    tokenizer: BaseTokenizer,
    threshold: float = 0.55,
    entropy_threshold: float = 1.5,
) -> ConfidenceReport:
    """End-to-end: log-probs -> transcript with per-word confidence."""
    tokens = token_confidences(log_probs, tokenizer)
    words = aggregate_words(tokens, threshold, entropy_threshold)
    text = tokenizer.decode([t.token_id for t in tokens])

    if tokens:
        # Geometric mean of token probabilities: length-normalised, and it
        # punishes a single near-zero token far harder than an arithmetic mean.
        log_sum = sum(math.log(max(t.probability, 1e-12)) for t in tokens)
        utt = math.exp(log_sum / len(tokens))
    else:
        utt = 0.0

    return ConfidenceReport(text=text, utterance_confidence=utt, words=words)
