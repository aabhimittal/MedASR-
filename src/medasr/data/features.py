"""Log-mel filterbank feature extraction.

Raw audio is a 1-D sequence of ~16,000 samples per second — far too long and
too redundant to feed a neural network directly. The standard ASR front-end
turns the waveform into a compact time-frequency picture:

    waveform ->  STFT  ->  power spectrogram  ->  mel filterbank  ->  log
      (T,)        (F, N)         (F, N)               (M, N)         (M, N)

* **STFT** slides a short window (25 ms) over the signal every 10 ms and takes
  a Fourier transform of each window, giving a spectrum per frame.
* The **mel filterbank** is a set of ``M`` triangular filters spaced on the
  *mel* scale, which is roughly logarithmic and matches human pitch
  perception. It compresses hundreds of FFT bins into ~80 perceptually-spaced
  channels.
* The **log** compresses the huge dynamic range of audio power into something
  a network trains on comfortably (loudness in dB-like units).

The result is an ``(n_mels, num_frames)`` "image" — the input to the encoder.
This implementation is pure PyTorch (only ``torch.stft`` + a hand-built mel
matrix) so it has no hard dependency on torchaudio.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    # HTK mel scale, the convention most ASR toolkits use.
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Build the ``(n_mels, n_fft // 2 + 1)`` triangular mel filter matrix.

    Each row is one triangular filter; multiplying it against a power spectrum
    integrates the energy falling inside that mel band.
    """
    f_max = f_max or sample_rate / 2
    n_freqs = n_fft // 2 + 1

    # Frequencies of each FFT bin, in Hz.
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

    # n_mels + 2 equally-spaced points on the mel scale -> band edges.
    m_min, m_max = _hz_to_mel(torch.tensor(f_min)), _hz_to_mel(torch.tensor(f_max))
    m_points = torch.linspace(m_min.item(), m_max.item(), n_mels + 2)
    f_points = _mel_to_hz(m_points)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = f_points[m - 1], f_points[m], f_points[m + 1]
        # Rising edge from left->center, falling edge from center->right.
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fb[m - 1] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb


class LogMelExtractor(nn.Module):
    """Waveform ``(..., T)`` -> log-mel features ``(..., n_mels, N)``.

    Registered as an ``nn.Module`` with the mel matrix and window as buffers so
    the whole front-end moves to GPU with ``.to(device)`` and is saved with the
    model state dict — guaranteeing train/inference feature parity.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        normalize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.normalize = normalize
        self.eps = eps

        self.register_buffer("window", torch.hann_window(self.win_length))
        self.register_buffer(
            "fb", mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)
        )

    def num_frames(self, num_samples: int) -> int:
        """Frames produced for a waveform of ``num_samples`` (center-padded STFT)."""
        return num_samples // self.hop_length + 1

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Complex STFT -> power spectrogram (F, N).
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)  # (B, F, N)

        # Apply mel filterbank: (M, F) @ (B, F, N) -> (B, M, N).
        mel = torch.matmul(self.fb, power)
        log_mel = torch.log(mel + self.eps)

        if self.normalize:
            # Per-utterance mean/var normalisation over time.
            mean = log_mel.mean(dim=-1, keepdim=True)
            std = log_mel.std(dim=-1, keepdim=True)
            log_mel = (log_mel - mean) / (std + self.eps)

        return log_mel.squeeze(0) if squeeze else log_mel
