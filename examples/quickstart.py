"""End-to-end quickstart in a single script (no CLI, no real dataset).

Run it directly::

    python examples/quickstart.py

It fabricates a handful of synthetic clinical utterances, builds a character
tokenizer, trains a *tiny* Conformer-CTC for a few epochs on CPU, and then
transcribes a clip. Because the audio is synthetic tones (see
``medasr.cli.synth_data``), the transcript will be garbage — the point is to
show how the pieces connect, not to produce a usable model. Swap in a real
16 kHz manifest and a full config to train something real.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from medasr.cli.synth_data import render_waveform, write_wav
from medasr.config import Config
from medasr.data.dataset import ASRDataset, collate_batch
from medasr.data.features import LogMelExtractor
from medasr.decoding import decode
from medasr.models.ctc import ConformerCTC
from medasr.tokenizer import CharTokenizer
from medasr.training import Trainer

SENTENCES = [
    "start metformin 500 mg twice daily",
    "no known drug allergies",
    "blood pressure is 130 over 85 today",
    "continue lisinopril 10 mg once daily",
]


def build_corpus(root: Path) -> Path:
    rows = []
    for i in range(8):
        text = SENTENCES[i % len(SENTENCES)]
        wav_path = root / f"u{i}.wav"
        write_wav(wav_path, render_waveform(text), 16000)
        rows.append({"audio_filepath": str(wav_path), "text": text})
    manifest = root / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    return manifest


def main() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = build_corpus(root)

        # 1. Config — tiny, CPU-friendly.
        cfg = Config()
        cfg.model.d_model = 64
        cfg.model.num_layers = 2
        cfg.model.num_heads = 2
        cfg.model.conv_kernel_size = 15
        cfg.training.batch_size = 4
        cfg.training.num_epochs = 5
        cfg.training.device = "cpu"
        cfg.training.num_workers = 0
        cfg.training.warmup_steps = 5
        cfg.experiment.output_dir = str(root / "run")

        # 2. Tokenizer from the transcripts.
        tokenizer = CharTokenizer.build(SENTENCES)
        print(f"Vocab size: {tokenizer.vocab_size}")

        # 3. Feature extractor + dataset + loader.
        fx = LogMelExtractor(n_mels=cfg.features.n_mels)
        ds = ASRDataset(manifest, tokenizer, fx)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_batch)

        # 4. Model + trainer.
        model = ConformerCTC(cfg.model, tokenizer.vocab_size, tokenizer.blank_id)
        print(f"Parameters: {model.num_parameters() / 1e6:.2f}M")
        trainer = Trainer(model, tokenizer, cfg, loader, loader)
        trainer.fit()

        # 5. Transcribe the first clip.
        feats, _, ref = ds[0]
        log_probs, out_len = model.transcribe_features(
            feats.unsqueeze(0), torch.tensor([feats.shape[-1]])
        )
        hyp = decode(log_probs, out_len, tokenizer)[0]
        print(f"\nREF: {ref}\nHYP: {hyp!r}  (expected to be poor on synthetic audio)")


if __name__ == "__main__":
    main()
