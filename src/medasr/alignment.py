"""CTC forced alignment: recover *when* each token was spoken.

CTC training deliberately marginalises over alignments — that is the trick
that lets it learn without frame labels. But at inference time we often want
the alignment back:

* a dictation UI must place the cursor and let the clinician click a word to
  replay that instant of audio;
* an editor needs to re-record a single misheard drug name, not the paragraph;
* quality tooling wants to know which words the model spent the fewest frames
  on (a strong correlate of errors).

Given the per-frame log-probs and a *known* token sequence, the Viterbi pass
below finds the single most likely frame path through the CTC lattice. The
lattice is the target interleaved with blanks::

    target:    c   a   t
    extended:  ␣ c ␣ a ␣ t ␣          (length 2U+1)

From state ``s`` at frame ``t`` you may:

* **stay** at ``s``;
* **advance** to ``s+1``;
* **skip** to ``s+2``, but only when that lands on a real token and it differs
  from the token two states back — otherwise two identical letters would
  collapse into one and "ll" in "penicillin" would be lost.

Frame indices are converted to seconds using the feature hop *and* the
encoder's subsampling factor, since one encoder frame covers several mel
frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch

from medasr.tokenizer import BaseTokenizer

NEG_INF = -1e30


@dataclass
class TokenSpan:
    """One aligned token and the frames/seconds it occupies."""

    token_id: int
    text: str
    start_frame: int
    end_frame: int          # exclusive
    start_s: float
    end_s: float
    score: float            # mean log-prob of the token over its frames


@dataclass
class WordSpan:
    """A whitespace-delimited word with its time span and confidence."""

    text: str
    start_s: float
    end_s: float
    score: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def frames_to_seconds(frame: int, hop_length: int, sample_rate: int,
                      subsampling_factor: int = 4) -> float:
    """Convert an *encoder* frame index to seconds of wall-clock audio."""
    return frame * subsampling_factor * hop_length / float(sample_rate)


def forced_align(
    log_probs: torch.Tensor,
    targets: Sequence[int],
    blank_id: int = 0,
) -> List[int]:
    """Viterbi-align ``targets`` to ``log_probs``.

    Args:
        log_probs: ``(T, V)`` per-frame log probabilities for one utterance.
        targets: the known token ids (no blanks).
        blank_id: index of the CTC blank.

    Returns:
        A list of length ``T`` giving, for each frame, the index into the
        extended (blank-interleaved) target sequence.

    Raises:
        ValueError: if the target cannot fit in ``T`` frames. This is the
            classic CTC infeasibility case and callers must handle it rather
            than receive a meaningless alignment.
    """
    T, _ = log_probs.shape
    U = len(targets)
    if U == 0:
        return [0] * T

    # Extended sequence: blank, t0, blank, t1, ..., blank  -> length 2U+1
    ext: List[int] = [blank_id]
    for t in targets:
        ext.append(int(t))
        ext.append(blank_id)
    S = len(ext)

    # Minimum frames needed: one per token plus a blank between equal neighbours.
    min_frames = U + sum(1 for i in range(1, U) if targets[i] == targets[i - 1])
    if T < min_frames:
        raise ValueError(
            f"Cannot align {U} tokens to {T} frames (need >= {min_frames}). "
            "The audio is too short for this transcript."
        )

    lp = log_probs.detach().to(torch.float32).cpu()
    dp = torch.full((T, S), NEG_INF, dtype=torch.float32)
    back = torch.zeros((T, S), dtype=torch.long)

    # Frame 0 may only start on the first blank or the first real token.
    dp[0, 0] = lp[0, ext[0]]
    if S > 1:
        dp[0, 1] = lp[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            best, best_prev = dp[t - 1, s], s               # stay
            if s > 0 and dp[t - 1, s - 1] > best:
                best, best_prev = dp[t - 1, s - 1], s - 1   # advance
            # Skip a blank, only onto a distinct real token.
            if (
                s > 1
                and ext[s] != blank_id
                and ext[s] != ext[s - 2]
                and dp[t - 1, s - 2] > best
            ):
                best, best_prev = dp[t - 1, s - 2], s - 2
            dp[t, s] = best + lp[t, ext[s]]
            back[t, s] = best_prev

    # Must finish on the last token or the trailing blank.
    last = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = [0] * T
    path[T - 1] = last
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])
    return path


def align_tokens(
    log_probs: torch.Tensor,
    targets: Sequence[int],
    tokenizer: BaseTokenizer,
    hop_length: int = 160,
    sample_rate: int = 16000,
    subsampling_factor: int = 4,
) -> List[TokenSpan]:
    """Align and return one :class:`TokenSpan` per target token."""
    path = forced_align(log_probs, targets, tokenizer.blank_id)
    lp = log_probs.detach().to(torch.float32).cpu()

    spans: List[TokenSpan] = []
    for u, tok in enumerate(targets):
        state = 2 * u + 1  # real tokens sit at odd extended indices
        frames = [t for t, s in enumerate(path) if s == state]
        if not frames:
            continue
        start, end = frames[0], frames[-1] + 1
        score = float(sum(lp[t, tok] for t in frames) / len(frames))
        spans.append(
            TokenSpan(
                token_id=int(tok),
                text=tokenizer.decode([int(tok)]),
                start_frame=start,
                end_frame=end,
                start_s=frames_to_seconds(start, hop_length, sample_rate, subsampling_factor),
                end_s=frames_to_seconds(end, hop_length, sample_rate, subsampling_factor),
                score=score,
            )
        )
    return spans


def group_words(spans: Sequence[TokenSpan], separator: str = " ") -> List[WordSpan]:
    """Group character-level spans into whitespace-delimited words.

    Word score is the **mean** token log-prob converted to a probability. We
    deliberately do not use the minimum here — that is what
    :mod:`medasr.confidence` exposes for safety-critical gating — but the mean
    is the better display value.
    """
    words: List[WordSpan] = []
    buf: List[TokenSpan] = []

    def flush() -> None:
        if not buf:
            return
        text = "".join(s.text for s in buf)
        if text.strip():
            mean_lp = sum(s.score for s in buf) / len(buf)
            words.append(
                WordSpan(
                    text=text,
                    start_s=buf[0].start_s,
                    end_s=buf[-1].end_s,
                    score=math.exp(mean_lp),
                )
            )
        buf.clear()

    for span in spans:
        if span.text == separator:
            flush()
        else:
            buf.append(span)
    flush()
    return words
