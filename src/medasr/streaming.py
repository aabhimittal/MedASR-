"""Streaming (chunked) recognition for live dictation.

Batch transcription assumes the whole clip is available. A dictation UI cannot
wait: the clinician expects words to appear as they speak. That is awkward for
a Conformer, because self-attention and the depthwise convolution are both
**non-causal** — every output frame legitimately depends on future audio.

The standard production compromise, implemented here, is **chunked processing
with context carry-over**:

```
      |<-- left ctx -->|<--- chunk --->|<-- right ctx -->|
audio ....................................................
                       ^^^^^^^^^^^^^^^^
                       only this region is emitted
```

Each step re-encodes a window containing the new chunk plus real audio on both
sides, then keeps only the encoder frames corresponding to the chunk itself.
The context frames exist purely to give the convolution and attention a valid
receptive field, so emitted frames are (nearly) identical to what full-utterance
decoding would produce — we trade compute for latency, not accuracy.

Two details that are easy to get wrong and that matter a great deal:

* **Right context costs latency.** A chunk cannot be emitted until
  ``right_context`` more audio has arrived. Algorithmic latency is
  ``chunk + right_context`` seconds, and that is the number a product owner
  actually cares about. It is exposed as :attr:`StreamingConfig.latency_s`.
* **CTC collapse spans chunk boundaries.** If a chunk ends mid-``l`` in
  "penicillin" and the next begins on the same ``l``, naive per-chunk decoding
  emits ``ll``. We carry the last emitted symbol across boundaries so the
  collapse rule is applied to the *stream*, not to each chunk.

:meth:`StreamingTranscriber.finalize` flushes the tail, where no right context
will ever arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from medasr.audio import prepare_waveform
from medasr.data.features import LogMelExtractor
from medasr.models.ctc import ConformerCTC
from medasr.tokenizer import BaseTokenizer


@dataclass
class StreamingConfig:
    """Windowing parameters, expressed in seconds of audio."""

    chunk_s: float = 1.0
    left_context_s: float = 2.0
    right_context_s: float = 0.5
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if self.chunk_s <= 0:
            raise ValueError("chunk_s must be positive.")
        if self.left_context_s < 0 or self.right_context_s < 0:
            raise ValueError("Context sizes must be non-negative.")

    @property
    def latency_s(self) -> float:
        """Algorithmic latency: how far behind live audio the output runs."""
        return self.chunk_s + self.right_context_s

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_s * self.sample_rate)

    @property
    def left_samples(self) -> int:
        return int(self.left_context_s * self.sample_rate)

    @property
    def right_samples(self) -> int:
        return int(self.right_context_s * self.sample_rate)


@dataclass
class StreamingResult:
    """Text produced by one step, plus the running transcript."""

    delta: str          # newly emitted text (may be empty)
    text: str           # full transcript so far
    is_final: bool = False


class StreamingTranscriber:
    """Feed audio chunks in, get incremental transcript out.

    The transcriber is stateful and single-stream. For concurrent sessions
    create one instance per stream — they are cheap, holding only a buffer and
    a few integers. The model itself is shared and used read-only.
    """

    def __init__(
        self,
        model: ConformerCTC,
        tokenizer: BaseTokenizer,
        feature_extractor: LogMelExtractor,
        config: Optional[StreamingConfig] = None,
        device: str = "cpu",
        subsampling_factor: int = 4,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.cfg = config or StreamingConfig(sample_rate=feature_extractor.sample_rate)
        self.device = torch.device(device)
        self.subsampling_factor = subsampling_factor
        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        """Clear all stream state, ready for a new dictation."""
        self._buffer = torch.zeros(0)
        self._consumed = 0        # samples already emitted as chunks
        self._text_parts: List[str] = []
        self._last_symbol: Optional[int] = None
        self._finalized = False

    @property
    def text(self) -> str:
        return "".join(self._text_parts)

    # -- feeding -------------------------------------------------------------
    def accept_waveform(self, chunk: torch.Tensor) -> StreamingResult:
        """Append audio and emit whatever is now decodable.

        Accepts any chunk size — audio is buffered internally, so a caller can
        push 20 ms device buffers or 5 s uploads without changing the config.
        """
        if self._finalized:
            raise RuntimeError("Stream already finalized; call reset() to reuse.")
        if chunk.dim() > 1:
            chunk = chunk.mean(dim=0)
        chunk = torch.nan_to_num(chunk.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self._buffer = torch.cat([self._buffer, chunk])

        delta_parts: List[str] = []
        # Emit while a full chunk *plus* its right context has arrived.
        while (
            self._buffer.numel() - self._consumed
            >= self.cfg.chunk_samples + self.cfg.right_samples
        ):
            delta_parts.append(self._emit_chunk(self.cfg.chunk_samples))

        delta = "".join(delta_parts)
        if delta:
            self._text_parts.append(delta)
        return StreamingResult(delta=delta, text=self.text)

    def finalize(self) -> StreamingResult:
        """Flush trailing audio that will never receive right context."""
        if self._finalized:
            return StreamingResult(delta="", text=self.text, is_final=True)

        delta_parts: List[str] = []
        remaining = self._buffer.numel() - self._consumed
        if remaining > 0:
            delta_parts.append(self._emit_chunk(remaining))

        self._finalized = True
        delta = "".join(delta_parts)
        if delta:
            self._text_parts.append(delta)
        return StreamingResult(delta=delta, text=self.text, is_final=True)

    # -- internals -----------------------------------------------------------
    def _emit_chunk(self, chunk_samples: int) -> str:
        """Encode one window and decode only its central chunk region."""
        start = self._consumed
        end = min(start + chunk_samples, self._buffer.numel())

        win_start = max(0, start - self.cfg.left_samples)
        win_end = min(self._buffer.numel(), end + self.cfg.right_samples)
        window = self._buffer[win_start:win_end]

        # Guard against a window too short for the conv stem.
        window, _ = prepare_waveform(
            window, self.cfg.sample_rate, self.feature_extractor.hop_length
        )

        feats = self.feature_extractor(window.to(self.device)).unsqueeze(0)
        lengths = torch.tensor([feats.shape[-1]], device=self.device)
        with torch.no_grad():
            log_probs, out_lengths = self.model(feats, lengths)

        # Map the chunk's sample range onto encoder frames.
        hop = self.feature_extractor.hop_length * self.subsampling_factor
        n_frames = int(out_lengths[0].item())
        lo = (start - win_start) // hop
        hi = -(-(end - win_start) // hop)  # ceil division
        lo = max(0, min(lo, n_frames))
        hi = max(lo, min(hi, n_frames))
        if hi <= lo:
            self._consumed = end
            return ""

        self._consumed = end
        return self._decode_region(log_probs[0, lo:hi])

    def _decode_region(self, log_probs: torch.Tensor) -> str:
        """Greedy-collapse a slice, continuing the stream's collapse state."""
        best = log_probs.argmax(dim=-1).tolist()
        emitted: List[int] = []
        for idx in best:
            if idx != self._last_symbol and idx != self.tokenizer.blank_id:
                emitted.append(idx)
            self._last_symbol = idx
        return self.tokenizer.decode(emitted) if emitted else ""


def transcribe_stream(
    transcriber: StreamingTranscriber,
    chunks,
) -> str:
    """Convenience: drive a transcriber over an iterable of chunks."""
    for chunk in chunks:
        transcriber.accept_waveform(chunk)
    return transcriber.finalize().text
