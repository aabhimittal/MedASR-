"""SpecAugment — data augmentation directly on the feature spectrogram.

Clinical speech corpora are small and expensive to label, so the model easily
overfits. SpecAugment (Park et al., 2019) is a cheap, extremely effective
regulariser that hides random stripes of the log-mel "image":

* **Frequency masking** zeroes out a band of mel channels — the model must
  cope with a missing part of the spectrum (like a muffled microphone).
* **Time masking** zeroes out a span of frames — the model must infer through
  a dropout of the audio (like a brief cough or dropout).

Masks are applied on the fly during training only. We mask to zero, which,
because features are mean/variance normalised, corresponds to the channel's
average value.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_masks: int = 2,
        freq_mask_width: int = 27,
        time_masks: int = 2,
        time_mask_ratio: float = 0.05,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_mask_width = freq_mask_width
        self.time_masks = time_masks
        self.time_mask_ratio = time_mask_ratio

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Mask a single ``(n_mels, num_frames)`` feature tensor in-place-safe.

        Only active in training mode; returns the input unchanged in ``eval``.
        """
        if not self.training:
            return feats

        feats = feats.clone()
        n_mels, n_frames = feats.shape[-2], feats.shape[-1]

        # -- frequency masks --
        for _ in range(self.freq_masks):
            w = int(torch.randint(0, self.freq_mask_width + 1, (1,)).item())
            if w == 0 or w >= n_mels:
                continue
            f0 = int(torch.randint(0, n_mels - w, (1,)).item())
            feats[..., f0 : f0 + w, :] = 0.0

        # -- time masks (width scales with utterance length) --
        max_t = max(1, int(self.time_mask_ratio * n_frames))
        for _ in range(self.time_masks):
            w = int(torch.randint(0, max_t + 1, (1,)).item())
            if w == 0 or w >= n_frames:
                continue
            t0 = int(torch.randint(0, n_frames - w, (1,)).item())
            feats[..., :, t0 : t0 + w] = 0.0

        return feats
