"""Conformer convolution module.

While self-attention captures long-range dependencies, a **depthwise
convolution** captures the fine-grained *local* structure of speech — formant
transitions, plosive bursts, short co-articulation effects — that live within
a few tens of milliseconds. Combining the two is the core idea of the
Conformer ("convolution-augmented Transformer").

The module, in order:

1. **LayerNorm** then a **pointwise conv** that doubles the channels, followed
   by a **GLU** gate (halving them again) — a learned, data-dependent gate on
   what information passes through.
2. A **depthwise conv** (one filter per channel, kernel 31) — this is where
   the local temporal modelling happens, cheaply.
3. **BatchNorm + Swish**, then a final **pointwise conv** back to ``d_model``.

Padding on the depthwise conv is ``(kernel-1)//2`` so the output length equals
the input length. A padding mask zeroes out invalid frames before the conv so
that padded positions cannot leak into real ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from medasr.models.feedforward import Swish


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd for 'same' padding"
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,  # depthwise: one filter per channel
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T) for conv1d

        mask = pad_mask.unsqueeze(1) if pad_mask is not None else None
        if mask is not None:
            # pad_mask: (B, T) True where padded -> zero those frames.
            x = x.masked_fill(mask, 0.0)

        x = self.pointwise_conv1(x)
        x = self.glu(x)

        if mask is not None:
            # Re-mask before the depthwise conv. This is essential, not
            # defensive: pointwise_conv1 has a *bias*, so padded frames are no
            # longer zero after it. The depthwise conv has a wide kernel and
            # would smear that bias back into the last real frames -- making a
            # transcript depend on whatever else happened to be in the batch.
            # Zeroing here matches what F.conv1d's own padding supplies when
            # the utterance is run alone, which is what makes batching safe.
            x = x.masked_fill(mask, 0.0)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        if mask is not None:
            # Keep padding clean for the residual add and the next block.
            x = x.masked_fill(mask, 0.0)

        return x.transpose(1, 2)  # back to (B, T, D)
