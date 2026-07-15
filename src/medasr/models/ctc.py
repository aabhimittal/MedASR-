"""Conformer + CTC: the full acoustic model.

**Why CTC?** Audio and text are sequences of very different, unknown-aligned
lengths — we have ~25 encoder frames per second but only a few characters. We
don't have a per-frame label telling us which character is spoken at frame
``t``. Connectionist Temporal Classification (Graves et al., 2006) solves this
by:

1. Adding a special ``<blank>`` symbol (id 0).
2. Letting the network emit a symbol *or* blank at every frame.
3. Defining a collapsing rule — remove repeated symbols, then remove blanks —
   that maps a frame-level path to a final transcript
   (``h-h-<b>-e-l-l-<b>-l-o`` -> ``hello``).
4. Summing the probability of *all* frame paths that collapse to the target
   text, via an efficient forward-backward dynamic program. This sum is
   differentiable, so we can train with plain cross-entropy-style gradients
   and never need a frame-level alignment.

The model is just ``ConformerEncoder`` followed by a linear projection to the
vocabulary and a log-softmax. ``torch.nn.CTCLoss`` implements the forward
algorithm.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medasr.config import ModelConfig
from medasr.models.encoder import ConformerEncoder


class ConformerCTC(nn.Module):
    def __init__(self, model_cfg: ModelConfig, vocab_size: int, blank_id: int = 0):
        super().__init__()
        self.cfg = model_cfg
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        self.encoder = ConformerEncoder(
            input_dim=model_cfg.input_dim,
            d_model=model_cfg.d_model,
            num_layers=model_cfg.num_layers,
            num_heads=model_cfg.num_heads,
            ffn_expansion=model_cfg.ffn_expansion,
            conv_kernel_size=model_cfg.conv_kernel_size,
            dropout=model_cfg.dropout,
        )
        self.classifier = nn.Linear(model_cfg.d_model, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    # -- forward -------------------------------------------------------------
    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor):
        """Return ``(log_probs, output_lengths)``.

        ``log_probs`` has shape ``(B, T', vocab_size)`` — the per-frame
        distribution over output symbols. ``output_lengths`` are the true
        (subsampled) frame counts, needed to mask padding in the CTC loss.
        """
        enc, out_lengths = self.encoder(features, feature_lengths)
        logits = self.classifier(enc)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, out_lengths

    # -- loss ----------------------------------------------------------------
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        output_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # CTCLoss wants time-major input: (T', B, vocab_size).
        log_probs_tbc = log_probs.transpose(0, 1)
        return self.ctc_loss(
            log_probs_tbc,
            targets,
            output_lengths,
            target_lengths,
        )

    # -- convenience ---------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def transcribe_features(self, features: torch.Tensor, feature_lengths: torch.Tensor):
        """Run the model in eval mode and return log-probs for decoding."""
        was_training = self.training
        self.eval()
        log_probs, out_lengths = self.forward(features, feature_lengths)
        if was_training:
            self.train()
        return log_probs, out_lengths
