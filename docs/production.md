# MedASR — Production Features & Edge-Case Behaviour

[`concepts.md`](concepts.md) explains how the model works. This document covers
what sits *around* the model to make it deployable: the safety layers, the
streaming path, and the failure modes we deliberately handle.

---

## 1. Why a transcript is not enough

A recogniser that returns only a string forces every downstream consumer to
assume it is correct. In clinical dictation that assumption is unsafe: the
model always outputs *something*, and it has no way to signal "I was guessing".

`Transcriber.transcribe_detailed()` returns a `DetailedTranscription` carrying
the four things a chart entry actually needs:

| Field | Answers |
|-------|---------|
| `text` | What was said |
| `words` | When each word was said (for cursor placement / replay) |
| `confidence` | How sure the model was, per word |
| `safety_flags` | What a human must confirm before signing |

`needs_review` folds these into one boolean a UI can gate on, and
`review_reasons` explains it in words. It is deliberately conservative — any
low-confidence word, any clinical flag, or unusable audio routes to review.

```python
result = transcriber.transcribe_detailed(waveform)
if result.needs_review:
    show_for_confirmation(result.text, result.review_reasons)
else:
    commit_to_chart(result.text)
```

---

## 2. Confidence: the weakest link, not the average

`medasr/confidence.py` scores each token by its posterior at the emitting
frame, then aggregates a word by its **minimum**, not its mean. A word is only
as trustworthy as its shakiest character — which is exactly where `15` becomes
`1.5`. An average would dilute that single bad character into a
reassuring-looking score.

We also track **entropy**. Probability alone misses a real failure mode: a
frame can put 0.5 on the right token while spreading the rest over ten
plausible rivals (genuinely uncertain) or over one (a clean two-way toss-up).
High entropy is a reliable "the model was guessing" signal, and either
condition alone is enough to flag a word.

---

## 3. Drug-name safety: correct the typos, flag the confusions

`medasr/lexicon.py` does two *opposite* things, and the distinction is the
whole point:

**Correct** near-misses that are not words: `metfarmin` → `metformin`. Safe,
because the input is not a real drug.

**Never correct** — only flag — pairs that are both real and acoustically
adjacent:

```
hydralazine (antihypertensive)  vs  hydroxyzine (antihistamine)
clonidine   (antihypertensive)  vs  klonopin    (benzodiazepine)
```

"Correcting" one of these silently changes the prescription. They live in
`CONFUSABLE_PAIRS` and are surfaced for human confirmation instead. The test
suite asserts this property holds for **every** registered pair.

The lexicon also refuses to guess when two candidates tie at the same edit
distance — an ambiguous correction between two drug names is precisely the
error we must not introduce.

Dose checks catch the other classic dictation hazard: implausible magnitudes,
and sub-unit decimals (`0.5 mg`) where a dropped point shifts the dose tenfold.

---

## 4. PHI redaction — and its honest limits

`medasr/phi.py` detects the structured HIPAA Safe Harbor identifiers that
regular expressions genuinely handle well: MRN, SSN, phone, email, URL, dates,
ages over 89, ZIP.

**It is not a complete de-identifier.** Rule-based redaction has a real recall
ceiling — it will not reliably catch free-text patient names, unusual
institutions, or identifiers spoken as words ("record number three four
seven"). Safe Harbor requires removing all eighteen identifier classes. Treat
this as a strong first pass and defence-in-depth, not as certification; a
validated NER system or expert determination is still required before data
leaves a covered environment.

Redaction is **off by default** — the treating clinician should see the real
text. Turn it on for logs, analytics, and training corpora. The API only ever
returns PHI *counts*, never the matched strings.

---

## 5. Streaming: latency you can actually quote

A Conformer is non-causal — attention and the depthwise convolution both
legitimately depend on future audio. `medasr/streaming.py` uses the standard
production compromise, chunked processing with context carry-over:

```
      |<-- left ctx -->|<--- chunk --->|<-- right ctx -->|
audio ....................................................
                       ^^^^^^^^^^^^^^^^
                       only this region is emitted
```

Context frames exist purely to give the convolution and attention a valid
receptive field, so emitted frames closely match full-utterance decoding. We
trade compute for latency, not accuracy.

Two things that are easy to get wrong:

* **Right context is latency.** A chunk cannot be emitted until that much more
  audio arrives. `StreamingConfig.latency_s` (`chunk + right_context`) is the
  number to quote to a product owner.
* **CTC collapse spans chunk boundaries.** If a chunk ends mid-`l` in
  "penicillin" and the next begins on the same `l`, naive per-chunk decoding
  emits `ll`. We carry the last emitted symbol across boundaries.

Chunk size at the *caller* is irrelevant — audio is buffered internally, so
20 ms device buffers and 5 s uploads produce identical output. This is asserted
by `test_chunk_size_independence`.

---

## 6. Two real bugs the edge-case suite caught

These are worth recording, because both were invisible in ordinary use.

### Padding changed the transcript

`ConformerConvModule` masked padded frames to zero on entry — but
`pointwise_conv1` has a **bias**, so those frames were no longer zero
afterwards. The wide depthwise convolution then smeared that bias back into the
last real frames. Running the same clip alone produced different numbers than
running it in a batch, because `F.conv1d`'s own padding supplies true zeros.

In production this means **a transcript depending on whatever else happened to
be in the batch** — non-deterministic output under dynamic batching. The fix
re-masks immediately before the depthwise conv; `test_padding_does_not_change_
a_short_utterance` now pins the invariant at exact equality.

### Clipping detection was destroyed by DC removal

Conditioning removed the DC offset before measuring clipping. A fully
rail-pinned (clipped) clip has a huge DC component, so de-biasing turned
obvious clipping into apparent silence and the warning never fired. Clipping is
a property of the audio *as recorded*, so it is now measured first.

A third, subtler one is documented rather than fixed in code: with
`zero_infinity=True`, CTC gives **zero loss and zero gradient** to targets
longer than the encoder output. A corpus with over-long transcripts trains to a
plausible-looking loss curve while learning nothing from those clips.
`compute_loss` now detects and warns (or raises with `strict=True`).

---

## 7. Edge-case behaviour reference

What the system does with input that is not a clean 16 kHz utterance:

| Input | Behaviour |
|-------|-----------|
| Empty audio (0 samples) | `AudioValidationError` (→ HTTP 400) |
| 3-sample clip | Padded to the encoder minimum; transcribes to empty |
| NaN / Inf samples | Replaced with silence, reported in `AudioReport` |
| Digital silence | Flagged `is_silent`, routed to review |
| Clipped audio | Flagged `was_clipped`, routed to review |
| DC offset | Removed, reported |
| Stereo (either orientation) | Downmixed to mono |
| 60 s dictation | Handled; relative positional encoding extends |
| Wrong sample rate | Rejected with a clear message (or resampled with `[audio]`) |
| Target longer than frames | Warned as zero-gradient; `strict=True` raises |
| Empty transcript in a batch | Collates correctly, no zero-width tensor |
| Mixed 60-frame / 2000-frame batch | Correct per-item lengths, all finite |
| Unseen characters | Mapped to `<unk>`, dropped on decode |
| Near-uniform posterior | Beam search stays finite |

Every row is an assertion in `tests/test_edge_cases.py`, `tests/test_audio.py`,
or `tests/test_streaming.py`.

---

## 8. Language-model fusion

`medasr/lm.py` provides a dependency-free character n-gram LM for **shallow
fusion** in beam search:

```
score = log P_acoustic + alpha * log P_lm + beta * |tokens|
```

`beta` is not optional polish. Every extra token multiplies in another
probability below 1, so without an insertion bonus the LM systematically
prefers *shorter* strings — and in a clinical note the word it drops might be
"no". `test_insertion_bonus_does_not_shorten_output` guards this.

Clinical notes are formulaic, so the prior is unusually strong here. Swap in
KenLM for scale: the `log_prob(context, char)` interface is all the beam
search requires.
