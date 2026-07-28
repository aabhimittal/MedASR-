import pytest
import torch

from medasr.config import ModelConfig
from medasr.data.features import LogMelExtractor
from medasr.decoding import greedy_decode
from medasr.models.ctc import ConformerCTC
from medasr.streaming import (
    StreamingConfig,
    StreamingTranscriber,
    transcribe_stream,
)
from medasr.tokenizer import CharTokenizer

CORPUS = ["patient denies chest pain", "start metformin 500 mg"]


def build(seed=0):
    torch.manual_seed(seed)
    tok = CharTokenizer.build(CORPUS)
    cfg = ModelConfig(
        input_dim=80, d_model=32, num_layers=1, num_heads=2,
        ffn_expansion=2, conv_kernel_size=7, dropout=0.0,
    )
    model = ConformerCTC(cfg, tok.vocab_size).eval()
    return model, tok, LogMelExtractor()


def make(cfg=None, seed=0):
    model, tok, fx = build(seed)
    return StreamingTranscriber(
        model, tok, fx,
        cfg or StreamingConfig(chunk_s=0.5, left_context_s=1.0, right_context_s=0.25),
    )


# -- configuration -----------------------------------------------------------
def test_latency_is_chunk_plus_right_context():
    cfg = StreamingConfig(chunk_s=0.8, right_context_s=0.4)
    assert cfg.latency_s == pytest.approx(1.2)


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        StreamingConfig(chunk_s=0)
    with pytest.raises(ValueError):
        StreamingConfig(right_context_s=-1)


# -- basic streaming ---------------------------------------------------------
def test_streams_and_finalizes():
    st = make()
    audio = torch.randn(16000 * 2) * 0.1
    for i in range(0, audio.numel(), 1600):
        st.accept_waveform(audio[i : i + 1600])
    result = st.finalize()
    assert result.is_final
    assert isinstance(result.text, str)


def test_text_accumulates_monotonically():
    """A streaming transcript may only ever grow -- never rewrite history."""
    st = make()
    audio = torch.randn(16000 * 3) * 0.1
    seen = ""
    for i in range(0, audio.numel(), 3200):
        text = st.accept_waveform(audio[i : i + 3200]).text
        assert text.startswith(seen)
        seen = text
    assert st.finalize().text.startswith(seen)


def test_delta_matches_text_growth():
    st = make()
    audio = torch.randn(16000 * 2) * 0.1
    prev = ""
    for i in range(0, audio.numel(), 1600):
        r = st.accept_waveform(audio[i : i + 1600])
        assert r.text == prev + r.delta
        prev = r.text


def test_chunk_size_independence():
    """Buffering must make the caller's chunk size irrelevant to the output."""
    audio = torch.randn(16000 * 2) * 0.1
    outputs = []
    for size in (160, 1600, 8000):
        st = make()
        for i in range(0, audio.numel(), size):
            st.accept_waveform(audio[i : i + size])
        outputs.append(st.finalize().text)
    assert outputs[0] == outputs[1] == outputs[2]


# -- edge cases --------------------------------------------------------------
def test_tiny_chunks_do_not_crash():
    st = make()
    for _ in range(50):
        st.accept_waveform(torch.randn(37) * 0.1)
    assert isinstance(st.finalize().text, str)


def test_single_sample_chunk():
    st = make()
    st.accept_waveform(torch.tensor([0.1]))
    assert isinstance(st.finalize().text, str)


def test_finalize_without_any_audio():
    st = make()
    r = st.finalize()
    assert r.is_final and r.text == ""


def test_finalize_is_idempotent():
    st = make()
    st.accept_waveform(torch.randn(16000) * 0.1)
    first = st.finalize()
    second = st.finalize()
    assert second.text == first.text
    assert second.delta == ""


def test_accept_after_finalize_raises():
    st = make()
    st.finalize()
    with pytest.raises(RuntimeError):
        st.accept_waveform(torch.randn(1600))


def test_reset_clears_state():
    st = make()
    st.accept_waveform(torch.randn(16000) * 0.1)
    st.finalize()
    st.reset()
    assert st.text == ""
    st.accept_waveform(torch.randn(1600) * 0.1)  # usable again


def test_nan_input_is_survived():
    st = make()
    bad = torch.full((16000,), float("nan"))
    st.accept_waveform(bad)
    assert isinstance(st.finalize().text, str)


def test_stereo_chunk_downmixed():
    st = make()
    st.accept_waveform(torch.randn(2, 8000) * 0.1)
    assert isinstance(st.finalize().text, str)


def test_silence_produces_no_crash():
    st = make()
    st.accept_waveform(torch.zeros(16000 * 2))
    assert isinstance(st.finalize().text, str)


def test_streams_are_independent():
    """Two concurrent sessions must not share collapse state."""
    model, tok, fx = build()
    cfg = StreamingConfig(chunk_s=0.5, left_context_s=1.0, right_context_s=0.25)
    a = StreamingTranscriber(model, tok, fx, cfg)
    b = StreamingTranscriber(model, tok, fx, cfg)
    audio = torch.randn(16000 * 2) * 0.1

    for i in range(0, audio.numel(), 1600):
        a.accept_waveform(audio[i : i + 1600])
    # b consumes the identical audio in one go afterwards.
    b.accept_waveform(audio)
    assert a.finalize().text == b.finalize().text


def test_transcribe_stream_helper():
    st = make()
    chunks = [torch.randn(1600) * 0.1 for _ in range(10)]
    assert isinstance(transcribe_stream(st, chunks), str)


# -- consistency with batch mode --------------------------------------------
def test_streaming_output_is_a_plausible_subset_of_batch():
    """Streaming and batch decoding should agree on the alphabet they emit."""
    model, tok, fx = build()
    audio = torch.randn(16000 * 2) * 0.1

    feats = fx(audio).unsqueeze(0)
    with torch.no_grad():
        lp, lens = model(feats, torch.tensor([feats.shape[-1]]))
    batch_text = greedy_decode(lp, lens, tok)[0]

    st = StreamingTranscriber(
        model, tok, fx,
        StreamingConfig(chunk_s=1.0, left_context_s=2.0, right_context_s=0.5),
    )
    st.accept_waveform(audio)
    stream_text = st.finalize().text

    assert set(stream_text) <= set(batch_text) | {" "}
