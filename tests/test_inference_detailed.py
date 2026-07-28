"""Integration tests for the rich transcription path."""

import torch

from medasr.config import Config
from medasr.data.features import LogMelExtractor
from medasr.inference import DetailedTranscription, Transcriber
from medasr.lexicon import SafetyFlag
from medasr.models.ctc import ConformerCTC
from medasr.tokenizer import CharTokenizer

CORPUS = ["start metformin 500 mg twice daily", "patient denies chest pain"]


def build(seed=0):
    torch.manual_seed(seed)
    cfg = Config()
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.conv_kernel_size = 7
    cfg.model.dropout = 0.0
    tok = CharTokenizer.build(CORPUS)
    model = ConformerCTC(cfg.model, tok.vocab_size)
    return Transcriber(model, tok, LogMelExtractor(), cfg)


def test_returns_detailed_object():
    d = build().transcribe_detailed(torch.randn(16000) * 0.1)
    assert isinstance(d, DetailedTranscription)
    assert isinstance(d.text, str)
    assert d.audio is not None
    assert d.confidence is not None


def test_model_is_in_eval_mode():
    """Inference in train mode would make results batch-dependent."""
    assert not build().model.training


def test_timestamps_are_ordered_and_within_audio():
    tr = build()
    seconds = 2.0
    d = tr.transcribe_detailed(torch.randn(int(16000 * seconds)) * 0.1)
    for w in d.words:
        assert 0.0 <= w.start_s <= w.end_s <= seconds + 0.5
    for x, y in zip(d.words, d.words[1:]):
        assert x.start_s <= y.start_s


def test_confidence_in_valid_range():
    d = build().transcribe_detailed(torch.randn(16000) * 0.1)
    assert 0.0 <= d.confidence.utterance_confidence <= 1.0
    for w in d.confidence.words:
        assert 0.0 <= w.probability <= 1.0


def test_silent_audio_is_routed_to_review():
    d = build().transcribe_detailed(torch.zeros(16000))
    assert d.audio.is_silent
    assert d.needs_review
    assert any("silent" in r.lower() for r in d.review_reasons)


def test_clipped_audio_is_routed_to_review():
    d = build().transcribe_detailed(torch.ones(16000))
    assert d.audio.was_clipped
    assert d.needs_review


def test_nan_audio_survives_and_is_reported():
    d = build().transcribe_detailed(torch.full((16000,), float("nan")))
    assert d.audio.had_nan
    assert isinstance(d.text, str)


def test_very_short_audio_does_not_crash():
    """A 3-sample clip must yield a result, not a shape error."""
    d = build().transcribe_detailed(torch.tensor([0.1, -0.1, 0.05]))
    assert isinstance(d.text, str)
    assert d.audio.was_padded


def test_toggles_disable_stages():
    tr = build()
    wav = torch.randn(16000) * 0.1
    d = tr.transcribe_detailed(
        wav, timestamps=False, confidence=False, apply_lexicon=False
    )
    assert d.words == []
    assert d.confidence is None
    assert d.corrections == []


def test_phi_redaction_is_off_by_default_and_optional():
    tr = build()
    wav = torch.randn(16000) * 0.1
    assert tr.transcribe_detailed(wav).phi is None
    assert tr.transcribe_detailed(wav, redact_phi=True).phi is not None


def test_safety_flags_drive_review():
    d = DetailedTranscription(
        text="hydralazine",
        safety_flags=[SafetyFlag("confusable", "hydralazine", "confirm drug")],
    )
    assert d.needs_review
    assert "confirm drug" in d.review_reasons


def test_clean_result_needs_no_review():
    assert not DetailedTranscription(text="ok").needs_review


def test_plain_and_detailed_agree_on_text():
    """The detailed path must not change the transcript when the lexicon is off."""
    tr = build()
    wav = torch.randn(16000) * 0.1
    plain = tr.transcribe_waveform(wav)
    detailed = tr.transcribe_detailed(wav, apply_lexicon=False)
    assert plain == detailed.text
