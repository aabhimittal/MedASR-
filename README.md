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
diagram and shapes.

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
├── decoding.py          Greedy + CTC prefix beam search
├── inference.py         Transcriber (checkpoint → transcript)
├── training.py          Trainer + Noam warmup schedule
├── data/                Features · SpecAugment · Dataset
├── models/              Subsampling · FFN · Attention · Conv · Encoder · CTC
└── cli/                 train · transcribe · evaluate · build_tokenizer · synth_data
app/api.py               FastAPI dictation endpoint
configs/                 Experiment YAML
docs/                    Conceptual guide + architecture reference
tests/                   Unit + end-to-end tests
```

---

## Development

```bash
pytest -q            # full suite (runs on CPU in seconds)
```

Tests cover config validation, normalisation, tokenizer round-trips, feature
shapes, the model forward/backward, both decoders, the metrics, and a full
data→train→decode→evaluate smoke test.

Continuous integration (`.github/workflows/ci.yml`) runs the suite on Python
3.9 and 3.11 plus a CLI smoke test on every push and PR.

---

## Roadmap

- [ ] RNN-Transducer / attention decoder for streaming with an internal LM
- [ ] Shallow-fusion external language model in beam search
- [ ] Word-level timestamps for dictation cursor placement
- [ ] Punctuation & truecasing post-processor
- [ ] Distillation to a smaller on-device model

## References & license

Conformer (Gulati 2020), CTC (Graves 2006), Transformer-XL (Dai 2019),
SpecAugment (Park 2019). Full citations in [`docs/concepts.md`](docs/concepts.md).

Licensed under the [MIT License](LICENSE).
