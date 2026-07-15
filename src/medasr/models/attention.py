"""Multi-head self-attention with relative positional encoding.

Self-attention lets every frame look at every other frame — it is what gives
the Conformer its **global** view of an utterance, complementing the local
convolution module. Plain attention is position-agnostic, so we add positional
information.

Conformer uses **relative** positional encoding (the Transformer-XL scheme,
Dai et al. 2019) rather than absolute positions. Relative encoding models
"how far apart" two frames are instead of "where each frame sits", which
generalises better to utterances longer than those seen in training — an
important property when a dictation can run for minutes.

The attention score between query ``i`` and key ``j`` decomposes into a
content term and a position term::

    score(i, j) = (q_i + u) · k_j   +   (q_i + v) · pos_{i-j}

where ``u`` and ``v`` are learned global bias vectors and ``pos`` is a
sinusoidal encoding of the relative distance. The ``_rel_shift`` trick turns a
matrix indexed by absolute position into one indexed by relative position
cheaply.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPositionalEncoding(nn.Module):
    """Sinusoidal encodings for relative positions ``+(L-1) .. -(L-1)``."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._pe: torch.Tensor | None = None
        self._extend(max_len)

    def _extend(self, length: int) -> None:
        if self._pe is not None and self._pe.size(1) >= 2 * length - 1:
            return
        # Positions run from +(length-1) down to -(length-1).
        pe_pos = torch.zeros(length, self.d_model)
        pe_neg = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe_pos[:, 0::2] = torch.sin(position * div)
        pe_pos[:, 1::2] = torch.cos(position * div)
        pe_neg[:, 0::2] = torch.sin(-1 * position * div)
        pe_neg[:, 1::2] = torch.cos(-1 * position * div)

        # Concatenate positive (reversed) and negative parts -> (2L-1, d).
        pe_pos = torch.flip(pe_pos, dims=[0]).unsqueeze(0)
        pe_neg = pe_neg[1:].unsqueeze(0)
        self._pe = torch.cat([pe_pos, pe_neg], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return relative positional encodings for a length-``T`` sequence."""
        self._extend(x.size(1))
        pe = self._pe.to(device=x.device, dtype=x.dtype)
        center = pe.size(1) // 2
        start = center - x.size(1) + 1
        end = center + x.size(1)
        return self.dropout(pe[:, start:end])


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model)

        # Learned global content/position bias (u and v in the paper).
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """Shift the position dimension so column indices become relative.

        Input ``x`` has shape ``(B, H, T, 2T-1)``; the output ``(B, H, T, T)``
        aligns each query row with distances ``0 .. -(T-1)``.
        """
        b, h, t1, t2 = x.shape
        zero_pad = torch.zeros((b, h, t1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(b, h, t2 + 1, t1)
        x = x_padded[:, :, 1:].view_as(x)
        return x[:, :, :, : t2 // 2 + 1]

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.linear_q(x).view(b, t, self.h, self.d_k)
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        q = q.transpose(1, 2)  # (B, H, T, d_k)

        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)

        # Content and position score terms (Transformer-XL decomposition).
        q_u = (q + self.pos_bias_u.unsqueeze(1))
        q_v = (q + self.pos_bias_v.unsqueeze(1))
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))          # (B,H,T,T)
        matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))          # (B,H,T,2T-1)
        matrix_bd = self._rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        if mask is not None:
            # mask: (B, 1, 1, T) with True at padded key positions.
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # Guard against all-masked rows producing NaNs.
        attn = torch.nan_to_num(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(b, t, self.h * self.d_k)
        return self.linear_out(out)
