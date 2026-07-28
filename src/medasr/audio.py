"""Robust audio validation and conditioning.

In a lab, audio arrives clean. In a hospital it does not. Dictation clips come
from phone handsets, headsets left on mute, browser recorders that emit float
NaNs on buffer underrun, and devices with a DC bias that silently destroys the
first mel channel. This module is the hardening layer between "bytes someone
uploaded" and "tensor the model may consume".

Every function is deliberately **total**: given any finite-or-not float tensor
it either returns something the model can process, or raises a typed
:class:`AudioValidationError` that a server can turn into a 4xx instead of a
500. Silently feeding a corrupt tensor into the network is never an option —
in a clinical setting a confidently-wrong transcript is worse than an error.

Design rules:

* **Never emit NaN/Inf downstream.** Log-mel of a NaN is a NaN, which becomes
  a NaN loss and a corrupted model. We fail loudly or sanitise explicitly.
* **Never emit a clip too short for the encoder.** Two stride-2 convolutions
  need a minimum number of frames or PyTorch raises a shape error deep in the
  stack. We pad up front, where the fix is obvious.
* **Report, don't hide.** Conditioning returns an :class:`AudioReport` so a
  caller can log that a clip was clipped, silent, or resampled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


class AudioValidationError(ValueError):
    """Raised when audio cannot be safely used for recognition."""


# Two stride-2 kernel-3 convs need >= 7 frames to produce >= 1 output frame.
# We keep a margin so the conformer's depthwise conv also has something to chew.
MIN_FEATURE_FRAMES = 16


@dataclass
class AudioReport:
    """What we observed and fixed while conditioning a clip."""

    num_samples: int
    duration_s: float
    peak: float
    rms: float
    is_silent: bool = False
    was_clipped: bool = False
    had_nan: bool = False
    had_dc_offset: bool = False
    was_padded: bool = False
    warnings: List[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.warnings.append(msg)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Downmix any ``(channels, T)`` or ``(T, channels)`` clip to ``(T,)``.

    Ambiguity is resolved by assuming the *smaller* dimension is channels,
    which holds for every realistic clip (nobody records 40,000 channels).
    """
    if waveform.dim() == 1:
        return waveform
    if waveform.dim() != 2:
        raise AudioValidationError(
            f"Expected 1-D or 2-D audio, got {waveform.dim()}-D tensor."
        )
    ch_dim = 0 if waveform.shape[0] <= waveform.shape[1] else 1
    return waveform.mean(dim=ch_dim)


def sanitize(waveform: torch.Tensor, report: Optional[AudioReport] = None) -> torch.Tensor:
    """Replace non-finite samples with zeros.

    A single NaN anywhere in the clip poisons the STFT for *every* frame that
    its window touches, and then the whole utterance through normalisation.
    Zeroing is the conservative repair: it reads as a moment of silence.
    """
    finite = torch.isfinite(waveform)
    if bool(finite.all()):
        return waveform
    if report is not None:
        report.had_nan = True
        n_bad = int((~finite).sum().item())
        report.add(f"replaced {n_bad} non-finite sample(s) with silence")
    return torch.where(finite, waveform, torch.zeros_like(waveform))


def remove_dc_offset(waveform: torch.Tensor, report: Optional[AudioReport] = None,
                     threshold: float = 1e-3) -> torch.Tensor:
    """Subtract the mean if the clip carries a meaningful DC bias.

    Cheap USB interfaces often add a constant offset. It contributes a large
    0 Hz component that dominates the lowest mel filter and shifts the
    per-utterance normalisation statistics.
    """
    mean = waveform.mean()
    if abs(float(mean)) < threshold:
        return waveform
    if report is not None:
        report.had_dc_offset = True
        report.add(f"removed DC offset of {float(mean):+.4f}")
    return waveform - mean


def pad_to_min_length(
    waveform: torch.Tensor,
    hop_length: int,
    min_frames: int = MIN_FEATURE_FRAMES,
    report: Optional[AudioReport] = None,
) -> torch.Tensor:
    """Right-pad with silence so the encoder always has enough frames.

    A 40 ms cough would otherwise crash the convolutional stem with an opaque
    shape error. Padding turns a hard failure into an (almost certainly empty)
    transcript, which is the correct behaviour for a server.
    """
    needed_samples = max(0, (min_frames - 1) * hop_length + 1)
    if waveform.numel() >= needed_samples:
        return waveform
    pad = needed_samples - waveform.numel()
    if report is not None:
        report.was_padded = True
        report.add(f"padded {pad} sample(s) to reach {min_frames} feature frames")
    return torch.nn.functional.pad(waveform, (0, pad))


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """Scale so the loudest sample sits at ``target_peak``; silence unchanged."""
    peak = float(waveform.abs().max()) if waveform.numel() else 0.0
    if peak <= 0.0:
        return waveform
    return waveform * (target_peak / peak)


def detect_clipping(waveform: torch.Tensor, threshold: float = 0.999,
                    min_ratio: float = 1e-4) -> bool:
    """True when a non-trivial fraction of samples sits at full scale.

    Clipped input is a real accuracy risk (harmonic distortion smears the
    spectrum), so we surface it rather than silently transcribing it.
    """
    if waveform.numel() == 0:
        return False
    at_rail = (waveform.abs() >= threshold).sum().item()
    return (at_rail / waveform.numel()) > min_ratio


def rms(waveform: torch.Tensor) -> float:
    if waveform.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(waveform.to(torch.float64) ** 2)))


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: float = 20.0,
    threshold_db: float = -45.0,
    keep_pad_ms: float = 100.0,
) -> torch.Tensor:
    """Trim leading/trailing near-silence using a simple energy gate.

    Clinicians commonly start recording, gather their thoughts, then speak.
    Trimming those dead seconds is free latency and free accuracy (the encoder
    stops spending attention on room tone). We keep a short pad so the first
    phoneme is never clipped off — that is how you lose the "no" in "no known
    allergies", which inverts the clinical meaning.
    """
    if waveform.numel() == 0:
        return waveform
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = waveform.numel() // frame
    if n_frames < 2:
        return waveform

    usable = waveform[: n_frames * frame].view(n_frames, frame)
    energy = usable.to(torch.float64).pow(2).mean(dim=1).clamp_min(1e-12)
    ref = float(energy.max())
    if ref <= 1e-12:
        return waveform  # entirely silent; leave it to the caller
    db = 10.0 * torch.log10(energy / ref)
    voiced = (db > threshold_db).nonzero().flatten()
    if voiced.numel() == 0:
        return waveform

    pad = int(sample_rate * keep_pad_ms / 1000.0)
    start = max(0, int(voiced[0].item()) * frame - pad)
    end = min(waveform.numel(), (int(voiced[-1].item()) + 1) * frame + pad)
    return waveform[start:end]


def prepare_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_length: int = 160,
    *,
    trim: bool = False,
    normalize_peak: bool = False,
    min_frames: int = MIN_FEATURE_FRAMES,
    silence_rms: float = 1e-5,
) -> tuple[torch.Tensor, AudioReport]:
    """Full conditioning pipeline: raw tensor -> model-safe tensor + report.

    Order matters. We downmix and sanitise before measuring anything (stats on
    NaNs are meaningless), de-bias before trimming (a DC offset defeats the
    energy gate), and pad last so nothing can shrink the clip afterwards.
    """
    if waveform.numel() == 0:
        raise AudioValidationError("Audio is empty (0 samples).")
    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")

    wav = to_mono(waveform).to(torch.float32)
    report = AudioReport(
        num_samples=int(wav.numel()),
        duration_s=float(wav.numel()) / sample_rate,
        peak=0.0,
        rms=0.0,
    )

    wav = sanitize(wav, report)

    # Clipping must be measured on the signal *as recorded*, before any gain or
    # bias correction. A fully rail-pinned clip has a huge DC component, so
    # de-biasing it first would turn obvious clipping into apparent silence.
    report.was_clipped = detect_clipping(wav)
    if report.was_clipped:
        report.add("input appears clipped; transcription accuracy may degrade")

    wav = remove_dc_offset(wav, report)

    if trim:
        wav = trim_silence(wav, sample_rate)
    if normalize_peak:
        wav = peak_normalize(wav)

    report.peak = float(wav.abs().max()) if wav.numel() else 0.0
    report.rms = rms(wav)
    if report.rms < silence_rms:
        report.is_silent = True
        report.add("input is effectively silent")

    wav = pad_to_min_length(wav, hop_length, min_frames, report)
    # Belt and braces: conditioning must never emit non-finite values.
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav, report
