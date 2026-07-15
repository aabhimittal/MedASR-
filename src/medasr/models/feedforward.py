"""Position-wise feed-forward module (the Macaron FFN).

A Conformer block sandwiches its attention and convolution between *two*
feed-forward modules, each contributing a **half-step** residual
(``x + 0.5 * FFN(x)``). This "Macaron" structure — two thin bread halves
around the filling — was found to outperform the single post-attention FFN of
a vanilla Transformer.

Each FFN expands the width by ``expansion`` (typically 4x), applies the Swish
non-linearity, and projects back. Swish (``x * sigmoid(x)``) is smoother than
ReLU and is the activation used throughout the Conformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
