"""High-level inference: checkpoint + audio -> transcript.

This wraps the moving parts (feature extractor, model, tokenizer, decoder) into
one object so that application code — a CLI, a REST endpoint, a notebook —
never has to reassemble the pipeline by hand. It guarantees that the exact
front-end used in training is reused at inference, which is the single most
common source of accuracy regressions in deployed ASR.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

from medasr.config import Config
from medasr.data.dataset import load_audio
from medasr.data.features import LogMelExtractor
from medasr.decoding import decode
from medasr.models.ctc import ConformerCTC
from medasr.tokenizer import BaseTokenizer, load_tokenizer


class Transcriber:
    def __init__(
        self,
        model: ConformerCTC,
        tokenizer: BaseTokenizer,
        feature_extractor: LogMelExtractor,
        cfg: Config,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor.to(self.device)
        self.cfg = cfg

    # -- construction from disk ---------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str = "cpu",
    ) -> "Transcriber":
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = Config.from_dict(ckpt["config"])
        tokenizer = load_tokenizer(tokenizer_path)

        model = ConformerCTC(cfg.model, tokenizer.vocab_size, tokenizer.blank_id)
        model.load_state_dict(ckpt["model"])

        fx = LogMelExtractor(
            sample_rate=cfg.features.sample_rate,
            n_mels=cfg.features.n_mels,
            n_fft=cfg.features.n_fft,
            hop_length=cfg.features.hop_length,
            win_length=cfg.features.win_length,
            f_min=cfg.features.f_min,
            f_max=cfg.features.f_max,
            normalize=cfg.features.normalize,
        )
        return cls(model, tokenizer, fx, cfg, device)

    # -- transcription -------------------------------------------------------
    @torch.no_grad()
    def transcribe_waveform(self, waveform: torch.Tensor) -> str:
        waveform = waveform.to(self.device)
        feats = self.feature_extractor(waveform)          # (n_mels, T)
        feats = feats.unsqueeze(0)                          # (1, n_mels, T)
        lengths = torch.tensor([feats.shape[-1]], device=self.device)
        log_probs, out_lengths = self.model(feats, lengths)
        return decode(
            log_probs,
            out_lengths,
            self.tokenizer,
            self.cfg.decoding.strategy,
            self.cfg.decoding.beam_size,
        )[0]

    def transcribe_file(self, audio_path: str | Path) -> str:
        wav = load_audio(str(audio_path), self.feature_extractor.sample_rate)
        return self.transcribe_waveform(wav)
