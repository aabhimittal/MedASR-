"""Convolutional subsampling front-end.

Log-mel features arrive at 100 frames/second. Running 16 self-attention layers
at that rate is wasteful (attention cost grows with the square of the sequence
length) and unnecessary — speech does not change meaningfully every 10 ms.

Two stacked 2-D convolutions with stride 2 downsample the time axis by 4x
(100 -> 25 frames/second) *and* project the mel channels into the model width.
This is the standard Conformer/Transformer-ASR input stem. It also gives the
network a small local receptive field before global attention takes over.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv2dSubsampling(nn.Module):
    """(B, T, n_mels) -> (B, T//4, d_model), with matching length arithmetic."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # After two stride-2, kernel-3 convs the frequency dim shrinks as below.
        subsampled_freq = ((input_dim - 1) // 2 - 1) // 2
        self.out = nn.Linear(d_model * subsampled_freq, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def subsampled_length(lengths: torch.Tensor) -> torch.Tensor:
        """Map input frame counts to output frame counts (same conv arithmetic)."""
        lengths = (lengths - 1) // 2
        lengths = (lengths - 1) // 2
        return lengths.clamp(min=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (B, T, F) -> add channel dim -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = self.conv(x)  # (B, C, T', F')
        b, c, t, f = x.shape
        # Merge channel + frequency, project to d_model.
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.dropout(self.out(x))
        return x, self.subsampled_length(lengths)
