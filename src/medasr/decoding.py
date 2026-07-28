"""CTC decoders: turn per-frame log-probs into text.

The model outputs a ``(T, vocab)`` distribution per utterance. To read a
transcript we must pick a path and collapse it under the CTC rule (remove
repeats, then remove blanks). Two strategies:

* **Greedy / best-path** — take the arg-max symbol at each frame, then
  collapse. Fast and, for a well-trained model, usually within a fraction of a
  percent WER of the optimum. This is the default.

* **Prefix beam search** — greedy commits to one symbol per frame and can be
  fooled because many paths collapse to the same text. Prefix beam search
  keeps the top-``k`` *collapsed prefixes* and correctly sums the probability
  mass of every path leading to each prefix (tracking blank/non-blank endings
  separately). It is slower but higher quality, and it is the natural place to
  fuse an external language model — e.g. a model of clinical phrasing.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Sequence

import torch

from medasr.tokenizer import BaseTokenizer


# ---------------------------------------------------------------------------
# Greedy / best-path decoding
# ---------------------------------------------------------------------------
def greedy_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: BaseTokenizer,
) -> List[str]:
    """Best-path decode a batch of ``(B, T, vocab)`` log-probs into strings."""
    preds = log_probs.argmax(dim=-1)  # (B, T)
    results = []
    for b in range(preds.size(0)):
        length = int(output_lengths[b].item())
        path = preds[b, :length].tolist()
        results.append(_collapse(path, tokenizer))
    return results


def _collapse(path: Sequence[int], tokenizer: BaseTokenizer) -> str:
    """Apply the CTC collapse rule: merge repeats, drop blanks."""
    collapsed: List[int] = []
    prev = None
    for idx in path:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    ids = [i for i in collapsed if i != tokenizer.blank_id]
    return tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# Prefix beam search
# ---------------------------------------------------------------------------
def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: BaseTokenizer,
    beam_size: int = 8,
    lm=None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
) -> List[str]:
    """CTC prefix beam search over a batch. Pure-Python, CPU-side.

    For each prefix we track two scores: ``p_b`` (paths ending in blank) and
    ``p_nb`` (paths ending in a real symbol). Keeping them separate is what
    makes the repeat-collapse bookkeeping correct.

    Passing ``lm`` (anything with ``log_prob(context, char)``, e.g.
    :class:`medasr.lm.CharNGramLM`) enables **shallow fusion**: the beam is
    ranked by ``acoustic + lm_weight * lm + insertion_bonus * length``. The
    bonus offsets the LM's bias toward shorter strings — without it, fusion
    tends to delete words, which in a clinical note can invert the meaning.
    """
    log_probs = log_probs.cpu()
    results = []
    for b in range(log_probs.size(0)):
        length = int(output_lengths[b].item())
        results.append(
            _beam_search_single(
                log_probs[b, :length], tokenizer, beam_size,
                lm, lm_weight, insertion_bonus,
            )
        )
    return results


def _beam_search_single(
    log_probs: torch.Tensor,
    tokenizer: BaseTokenizer,
    beam_size: int,
    lm=None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
) -> str:
    blank = tokenizer.blank_id
    use_lm = lm is not None and lm_weight != 0.0
    # Cache LM scores per prefix so each is computed once, not once per rank.
    lm_cache: dict = {(): 0.0}

    def lm_score(prefix: tuple) -> float:
        """Cumulative LM log-prob of a prefix, built incrementally."""
        if prefix in lm_cache:
            return lm_cache[prefix]
        parent = lm_score(prefix[:-1])
        context = tokenizer.decode(list(prefix[:-1]))
        char = tokenizer.decode([prefix[-1]])
        # Multi-char or empty pieces (BPE) score as a unit.
        delta = sum(lm.log_prob(context + char[:i], char[i]) for i in range(len(char)))
        lm_cache[prefix] = parent + delta
        return lm_cache[prefix]

    def rank(prefix: tuple, scores: tuple) -> float:
        total = _logsumexp(scores[0], scores[1])
        if use_lm:
            total += lm_weight * lm_score(prefix) + insertion_bonus * len(prefix)
        return total

    # beam maps prefix(tuple of ids) -> [p_blank, p_non_blank] in log space.
    beam = {(): (0.0, -math.inf)}

    T, V = log_probs.shape
    for t in range(T):
        next_beam = defaultdict(lambda: (-math.inf, -math.inf))
        # Consider only the most probable symbols this frame for speed.
        topk = torch.topk(log_probs[t], min(beam_size * 2, V)).indices.tolist()
        if blank not in topk:
            topk.append(blank)

        for prefix, (p_b, p_nb) in beam.items():
            p_total = _logsumexp(p_b, p_nb)
            for s in topk:
                p = float(log_probs[t, s])
                if s == blank:
                    nb_b, nb_nb = next_beam[prefix]
                    nb_b = _logsumexp(nb_b, p_total + p)
                    next_beam[prefix] = (nb_b, nb_nb)
                    continue

                last = prefix[-1] if prefix else None
                if s == last:
                    # Repeat of the last symbol: extending needs a blank
                    # between them (so it uses p_b), while staying on the same
                    # prefix uses p_nb.
                    nb_b, nb_nb = next_beam[prefix]
                    next_beam[prefix] = (nb_b, _logsumexp(nb_nb, p_nb + p))

                    new_prefix = prefix + (s,)
                    e_b, e_nb = next_beam[new_prefix]
                    next_beam[new_prefix] = (e_b, _logsumexp(e_nb, p_b + p))
                else:
                    new_prefix = prefix + (s,)
                    e_b, e_nb = next_beam[new_prefix]
                    next_beam[new_prefix] = (e_b, _logsumexp(e_nb, p_total + p))

        # Prune to the top `beam_size` prefixes by fused score.
        scored = sorted(
            next_beam.items(), key=lambda kv: rank(kv[0], kv[1]), reverse=True
        )
        beam = dict(scored[:beam_size])

    best_prefix = max(beam.items(), key=lambda kv: rank(kv[0], kv[1]))[0]
    return tokenizer.decode(list(best_prefix))


def decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: BaseTokenizer,
    strategy: str = "greedy",
    beam_size: int = 8,
    lm=None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
) -> List[str]:
    """Dispatch to the requested decoding strategy."""
    if strategy == "greedy":
        return greedy_decode(log_probs, output_lengths, tokenizer)
    if strategy == "beam":
        return beam_search_decode(
            log_probs, output_lengths, tokenizer, beam_size,
            lm, lm_weight, insertion_bonus,
        )
    raise ValueError(f"Unknown decoding strategy: {strategy}")
