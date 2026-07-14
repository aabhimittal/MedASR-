"""The Conformer block and the full Conformer encoder.

A single **Conformer block** wires together the four modules built in the
sibling files, each as a residual connection::

    x = x + 0.5 * FFN_1(x)          # Macaron half-step feed-forward
    x = x +       MHSA(x)           # global context (relative self-attention)
    x = x +       Conv(x)           # local context (depthwise convolution)
    x = x + 0.5 * FFN_2(x)          # Macaron half-step feed-forward
    x = LayerNorm(x)

Stacking ``num_layers`` of these blocks on top of the convolutional
subsampling stem gives the encoder. The encoder turns an acoustic feature
sequence into a sequence of context-rich hidden vectors — one per (subsampled)
frame — which the CTC head then classifies into tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from medasr.models.attention import RelPositionalEncoding, RelPositionMultiHeadAttention
from medasr.models.convolution import ConformerConvModule
from medasr.models.feedforward import FeedForwardModule
from medasr.models.subsampling import Conv2dSubsampling


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """``(B,)`` lengths -> ``(B, max_len)`` boolean mask, True at padded frames."""
    device = lengths.device
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_expansion: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RelPositionMultiHeadAttention(d_model, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, attn_mask=None, pad_mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn_dropout(self.attn(self.attn_norm(x), pos_emb, attn_mask))
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 16,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling(input_dim, d_model, dropout)
        self.pos_encoding = RelPositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList(
            ConformerBlock(d_model, num_heads, ffn_expansion, conv_kernel_size, dropout)
            for _ in range(num_layers)
        )
        self.d_model = d_model

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        """``features`` is ``(B, n_mels, T)``; returns encodings + new lengths."""
        # Conv subsampling expects (B, T, n_mels).
        x = features.transpose(1, 2)
        x, lengths = self.subsampling(x, lengths)  # (B, T', d_model)

        pad_mask = lengths_to_padding_mask(lengths, x.size(1))  # (B, T')
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)          # (B, 1, 1, T')
        pos_emb = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x, pos_emb, attn_mask, pad_mask)
        return x, lengths
