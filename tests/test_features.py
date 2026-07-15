import torch

from medasr.data.augment import SpecAugment
from medasr.data.features import LogMelExtractor, mel_filterbank


def test_mel_filterbank_shape_and_nonneg():
    fb = mel_filterbank(n_mels=80, n_fft=400, sample_rate=16000)
    assert fb.shape == (80, 201)
    assert torch.all(fb >= 0)


def test_logmel_output_shape():
    fx = LogMelExtractor(n_mels=80, n_fft=400, hop_length=160)
    wav = torch.randn(16000)  # 1 second
    feats = fx(wav)
    assert feats.shape[0] == 80
    assert feats.shape[1] == fx.num_frames(16000)


def test_logmel_batched():
    fx = LogMelExtractor(n_mels=40)
    wav = torch.randn(4, 8000)
    feats = fx(wav)
    assert feats.shape[0] == 4
    assert feats.shape[1] == 40


def test_normalization_zero_mean():
    fx = LogMelExtractor(normalize=True)
    feats = fx(torch.randn(16000))
    # Per-channel mean over time should be ~0 after normalisation.
    assert torch.allclose(feats.mean(dim=-1), torch.zeros(feats.shape[0]), atol=1e-4)


def test_spec_augment_only_in_training():
    aug = SpecAugment(freq_masks=2, time_masks=2)
    feats = torch.ones(80, 200)
    aug.eval()
    assert torch.equal(aug(feats), feats)  # no-op in eval
    aug.train()
    out = aug(feats)
    assert out.shape == feats.shape
    assert (out == 0).any()  # something was masked
