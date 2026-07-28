# MedASR

**A Conformer-based medical speech-to-text model for clinical dictation.**

MedASR is a compact, from-scratch PyTorch implementation of a
[Conformer](https://arxiv.org/abs/2005.08100) acoustic encoder trained with
[CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) for transcribing
clinical dictation. It is built to be **read and understood**: every module is
small, documented, and paired with a step-by-step conceptual guide.

> ⚠️ **Not a medical device.** This is a research/education codebase. It ships
> with *synthetic* audio only, and the reference server is not configured for
> handling protected health information (PHI). Do not use it for real clinical
> decisions without appropriate validation, security, and regulatory review.

---

## Why Conformer + CTC for clinical dictation?

- **Conformer** combines self-attention (**global** context across a sentence)
  with convolution (**local** acoustic detail) — state-of-the-art for ASR.
- **CTC** learns the audio→text alignment on its own, so you only need
  `(audio, transcript)` pairs — no per-frame labels.
- **Character tokenizer** means no out-of-vocabulary problem: the model can
  spell any drug name whose letters it has seen.
- We report **CER as well as WER**, because in medicine a one-character slip
  ("15 mg" vs "1.5 mg", "hydralazine" vs "hydroxyzine") is a safety issue.

📖 **New here? Read the [step-by-step conceptual guide](docs/concepts.md)** —
it walks one utterance from microphone to transcript and explains every design
choice. See [`docs/architecture.md`](docs/architecture.md) for the data-flow
diagram and shapes, and [`docs/production.md`](docs/production.md) for the
safety layers, streaming, and edge-case behaviour.

---

## Beyond the transcript

A recogniser that returns only a string forces every consumer to assume it is
correct. MedASR returns what a chart entry actually needs:

```python
result = transcriber.transcribe_detailed(waveform)

result.text            # "start metformin 500 mg twice daily"
result.words           # per-word timings for cursor placement / replay
result.confidence      # per-word posterior + entropy
result.safety_flags    # look-alike drugs, implausible doses
result.needs_review    # one boolean a UI can gate on
result.review_reasons  # why, in words
```

| Feature | Module | What it protects against |
|---------|--------|--------------------------|
| **Word timestamps** | `alignment.py` | CTC forced alignment for click-to-replay |
| **Confidence + entropy** | `confidence.py` | Silent overconfidence on misheard words |
| **Drug-name safety** | `lexicon.py` | Look-alike drugs (`hydralazine`/`hydroxyzine`) |
| **PHI redaction** | `phi.py` | Identifiers leaking into logs and corpora |
| **Streaming** | `streaming.py` | Latency — live dictation from a non-causal model |
| **LM fusion** | `lm.py` | Acoustically plausible but nonsensical phrasing |
| **Audio hardening** | `audio.py` | NaNs, clipping, DC offset, sub-frame clips |

Two safety properties the test suite pins down:

* Look-alike drug pairs are **flagged, never auto-corrected** — silently
  swapping `hydralazine` for `hydroxyzine` changes the prescription.
* A word's confidence is its **weakest** character, not the average — that is
  exactly where `15 mg` becomes `1.5 mg`.

---

## Architecture at a glance

```
audio.wav
  │  log-mel features (80 × N)          medasr/data/features.py
  ▼
  │  conv subsampling (4× downsample)   medasr/models/subsampling.py
  ▼
  │  16 × Conformer block               medasr/models/encoder.py
  │     ½FFN → MHSA → Conv → ½FFN
  ▼
  │  linear + log-softmax → CTC         medasr/models/ctc.py
  ▼
  │  greedy / beam decode               medasr/decoding.py
  ▼
"start metformin 500 mg twice daily"
```

---

## Installation

```bash
git clone https://github.com/aabhimittal/medasr-.git
cd medasr-

# CPU PyTorch (see pytorch.org for a CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# The package + dev tools
pip install -e .[dev]

# Optional extras
pip install -e .[audio]   # torchaudio + soundfile (more formats, resampling)
pip install -e .[serve]   # FastAPI serving
pip install -e .[bpe]     # SentencePiece sub-word tokenizer
```

> Plain PCM WAVs load with **zero** extra audio dependencies (stdlib `wave`).
> Install `[audio]` for FLAC/MP3 and automatic resampling.

---

## Quickstart (60 seconds, no dataset)

Run the whole pipeline on fabricated data to see the pieces connect:

```bash
python examples/quickstart.py
```

Or drive it through the CLI:

```bash
# 1. Make a tiny synthetic corpus (synthetic tones, NOT real speech)
python -m medasr.cli.synth_data --out-dir data/synth --num 40

# 2. Build the character tokenizer from the transcripts
python -m medasr.cli.build_tokenizer \
    --manifests data/synth/train.jsonl data/synth/val.jsonl \
    --out artifacts/tokenizer.json

# 3. Train (override epochs for a quick smoke run)
python -m medasr.cli.train \
    --config configs/conformer_ctc.yaml \
    --train-manifest data/synth/train.jsonl \
    --val-manifest data/synth/val.jsonl \
    --tokenizer artifacts/tokenizer.json \
    --epochs 5

# 4. Evaluate (WER / CER) and transcribe
python -m medasr.cli.evaluate  --checkpoint runs/*/best.pt --tokenizer artifacts/tokenizer.json --manifest data/synth/val.jsonl
python -m medasr.cli.transcribe --checkpoint runs/*/best.pt --tokenizer artifacts/tokenizer.json data/synth/wavs/utt_0000.wav
```

The synthetic audio is meaningless tones, so accuracy will be poor — the point
is that data → train → decode → evaluate all work. Swap in a real dataset next.

---

## Training on real data

Point the trainer at a **JSON-lines manifest** (the NeMo/ESPnet convention):

```json
{"audio_filepath": "wavs/note_001.wav", "text": "patient denies chest pain"}
{"audio_filepath": "wavs/note_002.wav", "duration": 4.1, "text": "no known drug allergies"}
```

Requirements: **16 kHz mono** audio (or install `[audio]` to resample).
Everything else — model size, LR schedule, SpecAugment, decoding — is set in
[`configs/conformer_ctc.yaml`](configs/conformer_ctc.yaml), which maps 1:1 to
the dataclasses in `medasr/config.py`.

Suggested public corpora to start from (not medical, but good for pretraining):
LibriSpeech, Common Voice. For clinical text/audio, seek an IRB/BAA-approved
source; **never** commit PHI.

---

## Serving

```bash
export MEDASR_CHECKPOINT=runs/conformer_ctc_medical/best.pt
export MEDASR_TOKENIZER=artifacts/tokenizer.json
uvicorn app.api:app --host 0.0.0.0 --port 8000

curl -F "file=@clip.wav" http://localhost:8000/transcribe
# {"text": "start metformin 500 mg twice daily", "model_version": "0.1.0"}
```

---

## Project layout

```
src/medasr/
├── config.py            Typed, validated configuration
├── text.py              Medical text normalisation
├── tokenizer.py         Char / BPE tokenizers (blank at id 0)
├── metrics.py           WER + CER
├── decoding.py          Greedy + CTC prefix beam search (+ LM fusion)
├── inference.py         Transcriber + DetailedTranscription
├── training.py          Trainer + Noam warmup schedule
├── audio.py             Audio validation & conditioning
├── alignment.py         CTC forced alignment → word timestamps
├── confidence.py        Per-word confidence & entropy
├── lexicon.py           Drug-name correction + clinical safety flags
├── phi.py               PHI detection & redaction
├── streaming.py         Chunked streaming recognition
├── lm.py                Character n-gram LM for shallow fusion
├── data/                Features · SpecAugment · Dataset
├── models/              Subsampling · FFN · Attention · Conv · Encoder · CTC
└── cli/                 train · transcribe · evaluate · build_tokenizer · synth_data
app/api.py               FastAPI: /transcribe and /transcribe/detailed
configs/                 Experiment YAML
docs/                    Concepts · architecture · production guide
tests/                   Unit, integration, and industrial edge-case tests
```

---

## Development

```bash
pytest -q            # full suite (runs on CPU in seconds)
```

The suite covers the core pipeline (config, normalisation, tokenizer
round-trips, feature shapes, model forward/backward, both decoders, metrics,
and a data→train→decode→evaluate smoke test) plus **industrial edge cases**:
degenerate batches, padding invariance, CTC-infeasible targets, NaN/clipped/
silent audio, boundary-length inputs, determinism, and checkpoint fidelity.

Writing those tests found two real bugs, both invisible in ordinary use:

1. **Padding changed the transcript.** A bias in the conv module's pointwise
   layer leaked into padded frames, and the wide depthwise conv smeared it back
   into real audio — so output depended on what else was in the batch.
2. **Clipping detection was destroyed by DC removal**, which turned a
   rail-pinned clip into apparent silence before the check ran.

Both are described in [`docs/production.md`](docs/production.md#6-two-real-bugs-the-edge-case-suite-caught).

Continuous integration (`.github/workflows/ci.yml`) runs the suite on Python
3.9 and 3.11 plus a CLI smoke test on every push and PR.

---

## Roadmap

- [x] Shallow-fusion external language model in beam search
- [x] Word-level timestamps for dictation cursor placement
- [x] Per-word confidence scoring and human-review routing
- [x] Streaming (chunked) recognition
- [x] PHI redaction and drug-name safety checks
- [ ] RNN-Transducer / attention decoder for true streaming with an internal LM
- [ ] Punctuation & truecasing post-processor
- [ ] NER-based de-identification to complement the rule-based pass
- [ ] Distillation to a smaller on-device model

## References & license

Conformer (Gulati 2020), CTC (Graves 2006), Transformer-XL (Dai 2019),
SpecAugment (Park 2019). Full citations in [`docs/concepts.md`](docs/concepts.md).

Licensed under the [MIT License](LICENSE).
