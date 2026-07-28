"""Industrial edge cases for the core model and training path.

These are the failures that do not show up on a clean benchmark but do show up
in production: degenerate batches, silently-zero gradients, padding that
changes results, and inputs at the boundary of what the architecture can
process. Each test encodes a property we need to hold, not just an example
that happened to work.
"""

import math
import warnings

import pytest
import torch

from medasr.config import Config, ModelConfig
from medasr.data.dataset import Batch, collate_batch
from medasr.data.features import LogMelExtractor
from medasr.decoding import beam_search_decode, greedy_decode
from medasr.metrics import cer, wer
from medasr.models.ctc import ConformerCTC, count_infeasible, required_frames
from medasr.models.subsampling import Conv2dSubsampling
from medasr.tokenizer import CharTokenizer
from medasr.text import normalize


def small_model(vocab=20, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        input_dim=80, d_model=32, num_layers=2, num_heads=2,
        ffn_expansion=2, conv_kernel_size=7, dropout=0.0,
    )
    return ConformerCTC(cfg, vocab_size=vocab)


# ---------------------------------------------------------------------------
# CTC feasibility: the silent-zero-gradient trap
# ---------------------------------------------------------------------------
def test_required_frames_accounts_for_repeats():
    # "aab" needs a blank between the two a's -> 4 frames, not 3.
    t = torch.tensor([1, 1, 2])
    assert required_frames(t, 3) == 4
    # No repeats -> exactly one frame per token.
    assert required_frames(torch.tensor([1, 2, 3]), 3) == 3
    assert required_frames(torch.tensor([]), 0) == 0


def test_count_infeasible_detects_overlong_targets():
    targets = torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]])
    target_lengths = torch.tensor([4, 2])
    # First needs 4 frames but only has 2; second is fine.
    assert count_infeasible(torch.tensor([2, 10]), targets, target_lengths) == 1
    assert count_infeasible(torch.tensor([10, 10]), targets, target_lengths) == 0


def test_infeasible_target_warns_not_silently_zero():
    """The core trap: zero_infinity makes bad data cost nothing. Must warn."""
    model = small_model()
    feats = torch.randn(1, 80, 40)
    lengths = torch.tensor([40])
    log_probs, out_len = model(feats, lengths)

    n = int(out_len[0]) + 20  # guaranteed longer than the encoder output
    targets = torch.randint(1, 20, (1, n))
    target_lengths = torch.tensor([n])

    with pytest.warns(RuntimeWarning, match="zero gradient"):
        loss = model.compute_loss(log_probs, out_len, targets, target_lengths)
    assert torch.isfinite(loss)


def test_infeasible_target_can_raise_in_strict_mode():
    model = small_model()
    feats = torch.randn(1, 80, 40)
    log_probs, out_len = model(feats, torch.tensor([40]))
    n = int(out_len[0]) + 20
    with pytest.raises(ValueError, match="zero gradient"):
        model.compute_loss(
            log_probs, out_len, torch.randint(1, 20, (1, n)), torch.tensor([n]), strict=True
        )


def test_feasible_batch_does_not_warn():
    model = small_model()
    feats = torch.randn(2, 80, 160)
    log_probs, out_len = model(feats, torch.tensor([160, 160]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        model.compute_loss(
            log_probs, out_len, torch.randint(1, 20, (2, 5)), torch.tensor([5, 5])
        )


# ---------------------------------------------------------------------------
# Padding and batching invariants
# ---------------------------------------------------------------------------
def test_padding_does_not_change_a_short_utterance():
    """Batching a clip beside a longer one must not alter its transcript.

    This is the property that makes server-side dynamic batching safe. It only
    holds in eval mode -- BatchNorm in train mode mixes statistics across the
    batch, which is exactly why inference must never run in train mode.
    """
    model = small_model().eval()
    torch.manual_seed(3)
    short = torch.randn(80, 100)

    alone, len_alone = model(short.unsqueeze(0), torch.tensor([100]))

    padded = torch.zeros(2, 80, 300)
    padded[0, :, :100] = short
    padded[1] = torch.randn(80, 300)
    batched, len_batched = model(padded, torch.tensor([100, 300]))

    n = int(len_alone[0])
    assert int(len_batched[0]) == n
    assert torch.allclose(alone[0, :n], batched[0, :n], atol=1e-4)


def test_batch_order_does_not_matter():
    model = small_model().eval()
    torch.manual_seed(4)
    a, b = torch.randn(80, 200), torch.randn(80, 120)

    fwd = torch.zeros(2, 80, 200)
    fwd[0], fwd[1, :, :120] = a, b
    out_fwd, len_fwd = model(fwd, torch.tensor([200, 120]))

    rev = torch.zeros(2, 80, 200)
    rev[0, :, :120], rev[1] = b, a
    out_rev, len_rev = model(rev, torch.tensor([120, 200]))

    nb = int(len_fwd[1])
    assert torch.allclose(out_fwd[1, :nb], out_rev[0, :nb], atol=1e-4)


def test_extreme_length_variance_in_one_batch():
    model = small_model().eval()
    feats = torch.zeros(3, 80, 2000)
    lengths = torch.tensor([2000, 500, 60])
    for i, n in enumerate(lengths.tolist()):
        feats[i, :, :n] = torch.randn(80, n)
    log_probs, out_len = model(feats, lengths)
    assert torch.isfinite(log_probs).all()
    assert out_len[0] > out_len[1] > out_len[2]


def test_collate_handles_wildly_different_lengths():
    items = [
        (torch.randn(80, 5), torch.tensor([1, 2]), "ab"),
        (torch.randn(80, 500), torch.tensor([1]), "a"),
    ]
    batch = collate_batch(items)
    assert batch.features.shape == (2, 80, 500)
    assert batch.feature_lengths.tolist() == [5, 500]
    # Padding region must be exactly zero.
    assert torch.equal(batch.features[0, :, 5:], torch.zeros(80, 495))


def test_collate_with_empty_transcript():
    """An empty target must not produce a zero-width target tensor."""
    items = [
        (torch.randn(80, 50), torch.tensor([], dtype=torch.long), ""),
        (torch.randn(80, 50), torch.tensor([1, 2]), "ab"),
    ]
    batch = collate_batch(items)
    assert batch.target_lengths.tolist() == [0, 2]
    assert batch.targets.shape[1] == 2


def test_batch_to_device_is_a_noop_on_cpu():
    items = [(torch.randn(80, 10), torch.tensor([1]), "a")]
    batch = collate_batch(items).to("cpu")
    assert isinstance(batch, Batch)
    assert batch.features.device.type == "cpu"


# ---------------------------------------------------------------------------
# Boundary-length inputs
# ---------------------------------------------------------------------------
def test_minimum_viable_input_length():
    """Find the shortest input the conv stem accepts and confirm it works."""
    model = small_model().eval()
    ok = None
    for n in range(1, 40):
        if int(Conv2dSubsampling.subsampled_length(torch.tensor([n]))[0]) >= 1:
            try:
                lp, _ = model(torch.randn(1, 80, n), torch.tensor([n]))
                if torch.isfinite(lp).all():
                    ok = n
                    break
            except RuntimeError:
                continue
    assert ok is not None and ok <= 16, f"minimum usable length was {ok}"


def test_single_frame_length_is_clamped_not_negative():
    lengths = torch.tensor([1, 2, 3])
    out = Conv2dSubsampling.subsampled_length(lengths)
    assert (out >= 1).all()


def test_long_utterance_stays_finite():
    """A 60 s dictation must not overflow the relative-position encoding."""
    model = small_model().eval()
    n = 6000  # 60 s of 100 fps features
    lp, out_len = model(torch.randn(1, 80, n), torch.tensor([n]))
    assert torch.isfinite(lp).all()
    assert int(out_len[0]) > 1000


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------
def test_log_probs_are_normalised():
    model = small_model().eval()
    lp, _ = model(torch.randn(2, 80, 100), torch.tensor([100, 100]))
    total = lp.exp().sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)


def test_silent_input_does_not_produce_nan():
    """All-zero features are the normalisation worst case (zero variance)."""
    model = small_model().eval()
    lp, _ = model(torch.zeros(1, 80, 200), torch.tensor([200]))
    assert torch.isfinite(lp).all()


def test_extreme_magnitude_features_stay_finite():
    model = small_model().eval()
    for scale in (1e-8, 1e6):
        lp, _ = model(torch.randn(1, 80, 100) * scale, torch.tensor([100]))
        assert torch.isfinite(lp).all(), scale


def test_feature_extractor_never_emits_nan_on_silence():
    fx = LogMelExtractor(normalize=True)
    feats = fx(torch.zeros(16000))
    assert torch.isfinite(feats).all()


def test_beam_search_survives_degenerate_distributions():
    """A near-uniform posterior is the beam's worst case; must not produce inf."""
    tok = CharTokenizer.build(["abc"])
    uniform = torch.full((1, 10, tok.vocab_size), 0.0)
    lp = torch.log_softmax(uniform, dim=-1)
    out = beam_search_decode(lp, torch.tensor([10]), tok, beam_size=4)
    assert isinstance(out[0], str)


def test_decoders_ignore_padded_frames():
    tok = CharTokenizer.build(["abc"])
    torch.manual_seed(5)
    lp = torch.log_softmax(torch.randn(1, 50, tok.vocab_size), dim=-1)
    short = greedy_decode(lp, torch.tensor([5]), tok)[0]
    long = greedy_decode(lp, torch.tensor([50]), tok)[0]
    assert len(short) <= len(long)


def test_zero_length_output_decodes_to_empty():
    tok = CharTokenizer.build(["abc"])
    lp = torch.log_softmax(torch.randn(1, 10, tok.vocab_size), dim=-1)
    assert greedy_decode(lp, torch.tensor([0]), tok) == [""]


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------
def test_eval_forward_is_deterministic():
    model = small_model().eval()
    x, lengths = torch.randn(2, 80, 150), torch.tensor([150, 120])
    a, _ = model(x, lengths)
    b, _ = model(x, lengths)
    assert torch.equal(a, b)


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    model = small_model(seed=7).eval()
    x, lengths = torch.randn(1, 80, 120), torch.tensor([120])
    before, _ = model(x, lengths)

    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    restored = small_model(seed=99).eval()          # different init
    restored.load_state_dict(torch.load(path))
    after, _ = restored(x, lengths)

    assert torch.allclose(before, after, atol=1e-6)


def test_same_seed_gives_identical_models():
    a, b = small_model(seed=11), small_model(seed=11)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


# ---------------------------------------------------------------------------
# Text, tokenizer and metric edge cases
# ---------------------------------------------------------------------------
def test_metrics_on_empty_reference():
    assert wer([""], [""]) == 0.0
    assert cer([""], [""]) == 0.0
    assert wer([""], ["spurious words"]) == 0.0  # no reference words to divide by


def test_metrics_on_empty_hypothesis():
    assert wer(["two words"], [""]) == 1.0
    assert cer(["abc"], [""]) == 1.0


def test_metrics_mismatched_lengths_zip_shortest():
    assert wer(["a", "b"], ["a"]) == 0.0


def test_tokenizer_handles_unseen_characters():
    tok = CharTokenizer.build(["abc"])
    ids = tok.encode("abcxyz")
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == "abc"  # unknowns dropped on decode


def test_tokenizer_empty_string():
    tok = CharTokenizer.build(["abc"])
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_normalize_handles_unicode_and_empty():
    assert normalize("") == ""
    assert normalize("   ") == ""
    out = normalize("Ωmega café 5 milligrams")
    assert "cafe" in out and "mg" in out


def test_config_rejects_impossible_model():
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"d_model": 65, "num_heads": 4}})
