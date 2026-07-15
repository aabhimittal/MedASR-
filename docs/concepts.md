# MedASR — Step-by-Step Conceptual Guide

This document explains *why* every piece of the system exists and *how* it
works, following the path a single utterance takes from microphone to
transcript. Read it top to bottom and you will understand the whole model.

Each stage points at the file that implements it, so you can read the concept
and then the code side by side.

---

## 0. The problem: clinical dictation as sequence-to-sequence

A clinician speaks:

> "start metformin five hundred milligrams twice daily"

and we want the written note:

> `start metformin 500 mg twice daily`

Formally, **automatic speech recognition (ASR)** maps an audio signal
`x = (x_1, ..., x_T)` (tens of thousands of samples) to a text sequence
`y = (y_1, ..., y_U)` (tens of characters). The two sequences have *very
different lengths* and *no given alignment* — nobody labelled which audio
sample corresponds to which letter. Everything below is in service of learning
that mapping from `(audio, transcript)` pairs alone.

Medical dictation adds three twists that shape our design choices:

1. **Vocabulary** is specialised and unforgiving — drug names, dosages, anatomy.
2. **A one-character error can be clinically dangerous** ("15 mg" vs "1.5 mg"),
   so we track **CER**, not just WER.
3. **Data is scarce and sensitive** (PHI), so we lean hard on regularisation
   (SpecAugment) and never ship real audio in the repo.

---

## 1. From waveform to features — the log-mel spectrogram

*Implemented in `medasr/data/features.py`.*

Raw 16 kHz audio is 16,000 numbers per second — too long, too redundant, and
dominated by information the ear (and a phoneme) doesn't care about. The
classic front-end converts sound into a compact **time–frequency image**:

```
waveform ──STFT──► spectrogram ──mel filterbank──► mel spectrogram ──log──► features
  (T,)              (F, N)                          (80, N)                  (80, N)
```

- **STFT (Short-Time Fourier Transform):** slide a 25 ms window every 10 ms and
  take a Fourier transform of each window. This yields, for every 10 ms frame,
  how much energy sits at each frequency. 10 ms hop ⇒ **100 frames per second**.
- **Mel filterbank:** the ear resolves low frequencies finely and high
  frequencies coarsely. We collapse the hundreds of raw FFT bins into **80
  channels** spaced on the *mel* scale (perceptually uniform pitch). Each
  channel is a triangular filter integrating the energy in its band.
- **Log:** audio power spans many orders of magnitude. Taking the log turns it
  into something like decibels — a range a neural net trains on comfortably.
- **Per-utterance normalisation:** subtract mean / divide by std per channel so
  microphones and recording levels don't matter.

The output is an `(80, N)` matrix: the encoder's input. Think of it as a
grayscale image, 80 pixels tall, `N` frames wide.

---

## 2. Text normalisation and tokenisation — defining the labels

*Implemented in `medasr/text.py` and `medasr/tokenizer.py`.*

Before the model can learn `audio → text`, we must decide **what "text"
means**. If the same content appears as "5 mg", "5mg", and "five milligrams",
the model wastes capacity on orthographic noise. **Normalisation**
(`medasr/text.py`) canonicalises:

- spoken dictation commands → symbols ("period" → `.`),
- spelled-out units → abbreviations ("milligrams" → `mg`),
- accents, casing, and whitespace.

The same normaliser runs on **training labels and on model output before
scoring**, so we measure the model, not formatting differences.

Then a **tokenizer** turns normalised text into integer ids. We default to a
**character vocabulary**:

- No out-of-vocabulary problem — the model can spell any drug name whose
  letters it has seen.
- Tiny output layer, easy to train on small corpora.

**Index convention that the whole codebase depends on:** id `0` is the CTC
**blank** (see §5). Real characters occupy ids `1 …`. (`BPETokenizer` offers
sub-word units for larger corpora; it shifts SentencePiece ids up by one to
keep the blank at 0.)

---

## 3. The encoder input stem — convolutional subsampling

*Implemented in `medasr/models/subsampling.py`.*

Features arrive at 100 frames/second. Self-attention cost grows with the
**square** of sequence length, and speech doesn't change every 10 ms, so we
first **downsample time by 4×** with two stride-2 2-D convolutions. This:

- cuts the sequence to **25 frames/second** (4× cheaper attention),
- projects the 80 mel channels into the model width `d_model`,
- gives the network a small local receptive field before global attention.

The same convolution arithmetic is applied to the **length** of every utterance
(`subsampled_length`) so the model always knows how many frames are real versus
padding.

---

## 4. The Conformer block — convolution-augmented Transformer

*Implemented in `medasr/models/` (`feedforward.py`, `attention.py`,
`convolution.py`, `encoder.py`).*

The encoder is a stack of **Conformer blocks** (Gulati et al., 2020). The
insight: speech has both **global** structure (long-range dependencies, context
across a sentence) and **local** structure (formant transitions, plosives
within tens of ms). Transformers model the global well; convolutions model the
local well. A Conformer block does **both**, in this order:

```
x ← x + ½·FFN₁(x)          # Macaron feed-forward (half-step residual)
x ← x + MHSA(x)            # relative-position multi-head self-attention  ── global
x ← x + Conv(x)            # depthwise convolution module                 ── local
x ← x + ½·FFN₂(x)          # Macaron feed-forward (half-step residual)
x ← LayerNorm(x)
```

### 4a. Macaron feed-forward (`feedforward.py`)
Two thin feed-forward modules — one before, one after — each contributing a
**half-step** residual (`x + 0.5·FFN(x)`). Sandwiching the attention/conv
between two half-FFNs (the "Macaron" shape) beats the single FFN of a vanilla
Transformer. Activation is **Swish** (`x·sigmoid(x)`), smoother than ReLU.

### 4b. Relative multi-head self-attention (`attention.py`)
Attention lets every frame attend to every other frame — the model's **global**
view. Conformer uses **relative** positional encoding (Transformer-XL): it
models *how far apart* two frames are, not their absolute positions, which
generalises to utterances longer than any seen in training — vital for
minutes-long dictations. A padding **mask** prevents attending to padded
frames. The attention score splits into a content term and a position term;
the `_rel_shift` trick computes the position term efficiently.

### 4c. Convolution module (`convolution.py`)
A **depthwise** convolution (kernel 31, one filter per channel) captures local
temporal patterns cheaply. A **GLU** gate decides what passes through, and
BatchNorm + Swish stabilise training. A padding mask zeroes invalid frames so
padding can't bleed into real audio through the convolution.

Stacking `num_layers` (default 16) of these blocks is the **encoder**
(`encoder.py`): acoustic features in, a sequence of context-rich hidden vectors
out — one per subsampled frame.

---

## 5. The output head and loss — CTC

*Implemented in `medasr/models/ctc.py`.*

The encoder gives us ~25 hidden vectors per second; the transcript has a few
characters. We still have **no alignment** telling us which frame is which
letter. **Connectionist Temporal Classification (CTC)** solves this elegantly:

1. Add a special **`<blank>`** symbol (id 0) meaning "emit nothing / repeat".
2. A linear layer + log-softmax turns each frame's hidden vector into a
   distribution over `{blank} ∪ vocabulary`.
3. Define a **collapse rule**: merge consecutive duplicates, then drop blanks.
   So the frame path `h h <b> e l l <b> l o` collapses to `hello`.
4. Many frame paths collapse to the same text. CTC defines the probability of a
   transcript as the **sum over all paths** that collapse to it, computed by an
   efficient forward–backward dynamic program. This sum is differentiable, so
   we train with ordinary gradient descent — **no frame labels required**.

`torch.nn.CTCLoss` implements step 4. We pass it the true (unpadded) frame and
target lengths so padding contributes nothing.

Why CTC here (vs. an attention decoder or RNN-Transducer)? It is simple,
stable, **streaming-friendly**, and strong for dictation. The architecture
leaves room to add an attention or transducer decoder later; the encoder is the
same.

---

## 6. Decoding — from probabilities to text

*Implemented in `medasr/decoding.py`.*

At inference we have per-frame probabilities and want the single best
transcript.

- **Greedy / best-path:** take the arg-max symbol each frame and collapse.
  Fast, and usually within a fraction of a percent WER of optimal for a
  well-trained model. Default.
- **Prefix beam search:** greedy can be fooled because many paths collapse to
  the same text. Prefix beam search keeps the top-`k` *collapsed prefixes* and
  correctly sums the probability of every path reaching each prefix (tracking
  blank- vs non-blank-ending paths separately). Slower, higher quality, and the
  natural hook for an **external language model** of clinical phrasing.

---

## 7. Training dynamics — schedule, augmentation, checkpoints

*Implemented in `medasr/training.py` and `medasr/data/augment.py`.*

- **Noam / warmup schedule:** Conformers diverge if the learning rate starts
  high while LayerNorm and attention are still random. The LR **warms up**
  linearly for `warmup_steps`, then **decays** as `1/√step`.
- **SpecAugment:** with scarce clinical data the model overfits fast. We hide
  random **frequency bands** and **time spans** of the spectrogram during
  training — cheap, and one of the highest-value tricks in modern ASR.
- **Gradient clipping, accumulation, mixed precision** keep large-model
  training stable and fast on GPU (all in `Trainer`).
- **Checkpointing** saves model + optimizer + scheduler + config so runs
  resume, and tracks the **best WER** on validation.

---

## 8. Evaluation — WER and, crucially, CER

*Implemented in `medasr/metrics.py`.*

Both are edit distance normalised by reference length:

```
WER = (subs + inserts + deletes) / reference_words
```

WER splits on whitespace; **CER** on characters. For medicine CER is the metric
to watch: WER counts "hydralazine → hydroxyzine" as one whole-word error, the
same as any slip, while CER reveals it was a near-miss on a dangerous
look-alike drug. We report both.

---

## 9. Serving — the dictation endpoint

*Implemented in `app/api.py` and `medasr/inference.py`.*

`Transcriber` bundles the exact feature extractor, model, tokenizer, and
decoder from a checkpoint, guaranteeing **train/inference parity** (the #1 cause
of deployed-ASR regressions). `app/api.py` wraps it in a FastAPI `/transcribe`
endpoint. It is a *reference* deployment — real clinical use requires auth,
audit logging, encryption, and BAA-covered hosting for PHI.

---

## Putting it together

```
audio.wav
  │  LogMelExtractor                     (§1)
  ▼
(80, N) log-mel features
  │  Conv2dSubsampling  (4× downsample)  (§3)
  ▼
(N/4, d_model)
  │  16 × ConformerBlock  (FFN·MHSA·Conv·FFN)   (§4)
  ▼
(N/4, d_model) contextual encodings
  │  Linear + log-softmax                (§5)
  ▼
(N/4, vocab) per-frame log-probs
  │  greedy / beam decode + CTC collapse (§6)
  ▼
"start metformin 500 mg twice daily"
```

Training runs this forward, computes CTC loss against the tokenised transcript
(§5), and backpropagates — with the schedule and augmentation of §7 — until WER
and CER (§8) stop improving.

## References

- Gulati et al., *Conformer: Convolution-augmented Transformer for Speech
  Recognition*, 2020.
- Graves et al., *Connectionist Temporal Classification*, 2006.
- Dai et al., *Transformer-XL* (relative positional encoding), 2019.
- Park et al., *SpecAugment*, 2019.
- Vaswani et al., *Attention Is All You Need* (Noam schedule), 2017.
