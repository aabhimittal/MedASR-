import math

import pytest
import torch

from medasr.audio import (
    AudioValidationError,
    detect_clipping,
    pad_to_min_length,
    peak_normalize,
    prepare_waveform,
    remove_dc_offset,
    rms,
    sanitize,
    to_mono,
    trim_silence,
)


def tone(n=16000, freq=440, sr=16000, amp=0.3):
    t = torch.arange(n, dtype=torch.float32) / sr
    return amp * torch.sin(2 * math.pi * freq * t)


# -- mono conversion ---------------------------------------------------------
def test_to_mono_passthrough_1d():
    w = tone(1000)
    assert torch.equal(to_mono(w), w)


def test_to_mono_channels_first_and_last():
    stereo_cf = torch.stack([torch.ones(500), torch.zeros(500)])       # (2, 500)
    stereo_cl = stereo_cf.T.contiguous()                                # (500, 2)
    assert to_mono(stereo_cf).shape == (500,)
    assert to_mono(stereo_cl).shape == (500,)
    assert torch.allclose(to_mono(stereo_cf), torch.full((500,), 0.5))


def test_to_mono_rejects_3d():
    with pytest.raises(AudioValidationError):
        to_mono(torch.zeros(2, 2, 2))


# -- sanitisation ------------------------------------------------------------
def test_sanitize_replaces_nan_and_inf():
    w = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 0.5])
    out = sanitize(w)
    assert torch.isfinite(out).all()
    assert out[0] == 1.0 and out[4] == 0.5
    assert out[1] == 0.0


def test_sanitize_is_noop_when_clean():
    w = tone(100)
    assert torch.equal(sanitize(w), w)


# -- DC offset ---------------------------------------------------------------
def test_remove_dc_offset():
    w = tone(1000) + 0.4
    out = remove_dc_offset(w)
    assert abs(float(out.mean())) < 1e-5


def test_small_offset_left_alone():
    # Exactly zero-mean signal: must be returned untouched.
    w = torch.tensor([0.5, -0.5] * 500)
    assert torch.equal(remove_dc_offset(w), w)


# -- clipping ----------------------------------------------------------------
def test_detect_clipping_true_and_false():
    clipped = torch.ones(1000)
    assert detect_clipping(clipped)
    assert not detect_clipping(tone(1000, amp=0.3))


def test_detect_clipping_empty_is_false():
    assert not detect_clipping(torch.zeros(0))


# -- padding -----------------------------------------------------------------
def test_pad_to_min_length_extends_short_clip():
    w = torch.ones(10)
    out = pad_to_min_length(w, hop_length=160, min_frames=16)
    assert out.numel() >= 15 * 160 + 1
    assert torch.equal(out[:10], w)  # original preserved at the front


def test_pad_to_min_length_noop_when_long_enough():
    w = tone(16000)
    assert torch.equal(pad_to_min_length(w, 160, 16), w)


# -- normalisation & rms -----------------------------------------------------
def test_peak_normalize():
    out = peak_normalize(tone(1000, amp=0.01), target_peak=0.9)
    assert abs(float(out.abs().max()) - 0.9) < 1e-5


def test_peak_normalize_silence_unchanged():
    z = torch.zeros(100)
    assert torch.equal(peak_normalize(z), z)


def test_rms_of_silence_is_zero():
    assert rms(torch.zeros(100)) == 0.0
    assert rms(torch.zeros(0)) == 0.0


# -- silence trimming --------------------------------------------------------
def test_trim_silence_removes_leading_and_trailing():
    speech = tone(8000)
    padded = torch.cat([torch.zeros(16000), speech, torch.zeros(16000)])
    out = trim_silence(padded, 16000)
    assert out.numel() < padded.numel()
    assert out.numel() >= speech.numel()  # keeps a safety pad


def test_trim_silence_all_silent_returns_input():
    z = torch.zeros(8000)
    assert torch.equal(trim_silence(z, 16000), z)


def test_trim_silence_very_short_input():
    w = torch.ones(5)
    assert torch.equal(trim_silence(w, 16000), w)


# -- full pipeline -----------------------------------------------------------
def test_prepare_rejects_empty():
    with pytest.raises(AudioValidationError):
        prepare_waveform(torch.zeros(0), 16000)


def test_prepare_rejects_bad_sample_rate():
    with pytest.raises(AudioValidationError):
        prepare_waveform(tone(1000), 0)


def test_prepare_always_finite_and_long_enough():
    nasty = torch.tensor([float("nan"), float("inf"), 1e9, -1e9, 0.0])
    out, report = prepare_waveform(nasty, 16000, hop_length=160)
    assert torch.isfinite(out).all()
    assert out.numel() >= 15 * 160 + 1
    assert report.had_nan and report.was_padded


def test_prepare_flags_silence():
    out, report = prepare_waveform(torch.zeros(16000), 16000)
    assert report.is_silent
    assert torch.isfinite(out).all()


def test_prepare_flags_clipping():
    _, report = prepare_waveform(torch.ones(16000), 16000)
    assert report.was_clipped
    assert any("clipped" in w for w in report.warnings)


def test_prepare_report_fields():
    _, report = prepare_waveform(tone(16000), 16000)
    assert report.num_samples == 16000
    assert abs(report.duration_s - 1.0) < 1e-6
    assert report.peak > 0 and report.rms > 0
    assert not report.is_silent
