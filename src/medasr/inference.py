"""High-level inference: checkpoint + audio -> transcript.

This wraps the moving parts (feature extractor, model, tokenizer, decoder) into
one object so that application code — a CLI, a REST endpoint, a notebook —
never has to reassemble the pipeline by hand. It guarantees that the exact
front-end used in training is reused at inference, which is the single most
common source of accuracy regressions in deployed ASR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from medasr.alignment import WordSpan, align_tokens, group_words
from medasr.audio import AudioReport, prepare_waveform
from medasr.config import Config
from medasr.confidence import ConfidenceReport, score_utterance
from medasr.data.dataset import load_audio
from medasr.data.features import LogMelExtractor
from medasr.decoding import decode
from medasr.lexicon import Correction, MedicalLexicon, SafetyFlag
from medasr.models.ctc import ConformerCTC
from medasr.phi import RedactionReport, redact
from medasr.tokenizer import BaseTokenizer, load_tokenizer


@dataclass
class DetailedTranscription:
    """A transcript plus everything needed to review and act on it safely."""

    text: str
    audio: Optional[AudioReport] = None
    words: List[WordSpan] = field(default_factory=list)
    confidence: Optional[ConfidenceReport] = None
    corrections: List[Correction] = field(default_factory=list)
    safety_flags: List[SafetyFlag] = field(default_factory=list)
    phi: Optional[RedactionReport] = None

    @property
    def needs_review(self) -> bool:
        """True when a human should check this before it enters the record.

        Deliberately conservative: any low-confidence word, any clinical
        safety flag, or unusable audio all route the transcript to review.
        """
        if self.safety_flags:
            return True
        if self.confidence is not None and self.confidence.needs_review:
            return True
        if self.audio is not None and (self.audio.is_silent or self.audio.was_clipped):
            return True
        return False

    @property
    def review_reasons(self) -> List[str]:
        """Human-readable explanations for :attr:`needs_review`."""
        reasons: List[str] = []
        for flag in self.safety_flags:
            reasons.append(flag.message)
        if self.confidence is not None:
            low = [w.text for w in self.confidence.low_confidence_words]
            if low:
                reasons.append("Low-confidence word(s): " + ", ".join(low))
        if self.audio is not None:
            if self.audio.is_silent:
                reasons.append("Audio is effectively silent.")
            if self.audio.was_clipped:
                reasons.append("Audio is clipped; accuracy may be degraded.")
        return reasons


class Transcriber:
    def __init__(
        self,
        model: ConformerCTC,
        tokenizer: BaseTokenizer,
        feature_extractor: LogMelExtractor,
        cfg: Config,
        device: str = "cpu",
        lexicon: Optional[MedicalLexicon] = None,
        lm=None,
        lm_weight: float = 0.0,
        insertion_bonus: float = 0.0,
    ):
        self.device = torch.device(device)
        # eval() is not cosmetic here: BatchNorm in train mode would mix
        # statistics across a batch, so a transcript would depend on whatever
        # else was batched with it.
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor.to(self.device)
        self.cfg = cfg
        self.lexicon = lexicon or MedicalLexicon()
        self.lm = lm
        self.lm_weight = lm_weight
        self.insertion_bonus = insertion_bonus

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
    def _forward(self, waveform: torch.Tensor):
        """Condition audio and run the model. Returns (log_probs, lengths, report)."""
        wav, audio_report = prepare_waveform(
            waveform,
            self.feature_extractor.sample_rate,
            self.feature_extractor.hop_length,
        )
        feats = self.feature_extractor(wav.to(self.device)).unsqueeze(0)
        lengths = torch.tensor([feats.shape[-1]], device=self.device)
        log_probs, out_lengths = self.model(feats, lengths)
        return log_probs, out_lengths, audio_report

    def transcribe_waveform(self, waveform: torch.Tensor) -> str:
        log_probs, out_lengths, _ = self._forward(waveform)
        return decode(
            log_probs,
            out_lengths,
            self.tokenizer,
            self.cfg.decoding.strategy,
            self.cfg.decoding.beam_size,
            self.lm,
            self.lm_weight,
            self.insertion_bonus,
        )[0]

    def transcribe_file(self, audio_path: str | Path) -> str:
        wav = load_audio(str(audio_path), self.feature_extractor.sample_rate)
        return self.transcribe_waveform(wav)

    # -- rich transcription --------------------------------------------------
    def transcribe_detailed(
        self,
        waveform: torch.Tensor,
        *,
        timestamps: bool = True,
        confidence: bool = True,
        apply_lexicon: bool = True,
        redact_phi: bool = False,
    ) -> DetailedTranscription:
        """Transcribe and attach everything a clinical UI needs.

        This is the method a real dictation product calls. The plain string
        from :meth:`transcribe_waveform` is fine for a demo, but a chart entry
        needs to know *when* each word was said, *how sure* the model was, and
        whether anything requires human confirmation before it is signed.

        ``redact_phi`` is off by default: the clinician's own view should show
        the real text. Turn it on for logs, analytics, and training corpora.
        """
        log_probs, out_lengths, audio_report = self._forward(waveform)
        n = int(out_lengths[0].item())
        frame_log_probs = log_probs[0, :n].cpu()

        text = decode(
            log_probs, out_lengths, self.tokenizer,
            self.cfg.decoding.strategy, self.cfg.decoding.beam_size,
            self.lm, self.lm_weight, self.insertion_bonus,
        )[0]

        result = DetailedTranscription(text=text, audio=audio_report)

        if confidence:
            result.confidence = score_utterance(frame_log_probs, self.tokenizer)

        if timestamps and text:
            # Align against the decoded text to recover per-word timings.
            ids = self.tokenizer.encode(text)
            try:
                spans = align_tokens(
                    frame_log_probs, ids, self.tokenizer,
                    hop_length=self.cfg.features.hop_length,
                    sample_rate=self.cfg.features.sample_rate,
                    subsampling_factor=self.cfg.model.subsampling_factor,
                )
                result.words = group_words(spans)
            except ValueError:
                # Too few frames to align (very short clip): timings are simply
                # unavailable, which must not fail the whole transcription.
                result.words = []

        if apply_lexicon and text:
            lex = self.lexicon.apply(text)
            result.text = lex.text
            result.corrections = lex.corrections
            result.safety_flags = lex.flags

        if redact_phi:
            redaction = redact(result.text)
            result.text = redaction.redacted_text
            result.phi = redaction

        return result

    def transcribe_file_detailed(self, audio_path: str | Path, **kwargs):
        wav = load_audio(str(audio_path), self.feature_extractor.sample_rate)
        return self.transcribe_detailed(wav, **kwargs)
