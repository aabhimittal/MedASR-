"""End-to-end smoke test: data -> train step -> decode -> metrics.

Uses a couple of synthetic WAVs so the whole pipeline is exercised on CPU in
seconds. It does not assert on accuracy (a 2-step model learns nothing) — only
that the wiring runs and shapes line up.
"""

import json

import torch
from torch.utils.data import DataLoader

from medasr.cli.synth_data import render_waveform, write_wav
from medasr.config import Config
from medasr.data.dataset import ASRDataset, collate_batch
from medasr.data.features import LogMelExtractor
from medasr.models.ctc import ConformerCTC
from medasr.tokenizer import CharTokenizer
from medasr.training import Trainer


def _make_corpus(tmp_path, n=4):
    sentences = ["start metformin 500 mg", "no known drug allergies"]
    rows = []
    for i in range(n):
        text = sentences[i % len(sentences)]
        wav_path = tmp_path / f"u{i}.wav"
        write_wav(wav_path, render_waveform(text), 16000)
        rows.append({"audio_filepath": str(wav_path), "text": text})
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    return manifest, [r["text"] for r in rows]


def test_end_to_end_training_step(tmp_path):
    manifest, texts = _make_corpus(tmp_path)

    cfg = Config()
    # Shrink everything for a fast CPU test.
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.conv_kernel_size = 7
    cfg.training.batch_size = 2
    cfg.training.num_epochs = 1
    cfg.training.device = "cpu"
    cfg.training.num_workers = 0
    cfg.training.warmup_steps = 2
    cfg.experiment.output_dir = str(tmp_path / "run")

    tokenizer = CharTokenizer.build(texts)
    fx = LogMelExtractor(
        sample_rate=cfg.features.sample_rate,
        n_mels=cfg.features.n_mels,
        n_fft=cfg.features.n_fft,
        hop_length=cfg.features.hop_length,
    )

    ds = ASRDataset(manifest, tokenizer, fx)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_batch)

    model = ConformerCTC(cfg.model, tokenizer.vocab_size, tokenizer.blank_id)
    trainer = Trainer(model, tokenizer, cfg, loader, loader)

    loss_before = None
    for batch in loader:
        log_probs, out_len = model(batch.features, batch.feature_lengths)
        loss_before = model.compute_loss(
            log_probs, out_len, batch.targets, batch.target_lengths
        ).item()
        break
    assert loss_before is not None and loss_before > 0

    trainer.fit()  # one epoch, saves checkpoints
    assert (tmp_path / "run" / "epoch_1.pt").exists()

    metrics = trainer.evaluate()
    assert "wer" in metrics and "cer" in metrics
    assert 0.0 <= metrics["cer"]  # runs without error
