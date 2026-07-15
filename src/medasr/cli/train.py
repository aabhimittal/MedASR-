"""Train a Conformer-CTC model from a config file.

Usage::

    python -m medasr.cli.train \
        --config configs/conformer_ctc.yaml \
        --train-manifest data/synth/train.jsonl \
        --val-manifest data/synth/val.jsonl \
        --tokenizer artifacts/tokenizer.json
"""

from __future__ import annotations

import argparse
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from medasr.config import Config
from medasr.data.augment import SpecAugment
from medasr.data.dataset import ASRDataset, collate_batch
from medasr.data.features import LogMelExtractor
from medasr.models.ctc import ConformerCTC
from medasr.tokenizer import load_tokenizer
from medasr.training import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _AugmentingDataset(ASRDataset):
    """Wrap ASRDataset to apply SpecAugment to training features."""

    def __init__(self, *args, spec_augment: SpecAugment | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_augment = spec_augment

    def __getitem__(self, idx):
        feats, target, text = super().__getitem__(idx)
        if self.spec_augment is not None:
            self.spec_augment.train()
            feats = self.spec_augment(feats)
        return feats, target, text


def build_feature_extractor(cfg: Config) -> LogMelExtractor:
    f = cfg.features
    return LogMelExtractor(
        sample_rate=f.sample_rate,
        n_mels=f.n_mels,
        n_fft=f.n_fft,
        hop_length=f.hop_length,
        win_length=f.win_length,
        f_min=f.f_min,
        f_max=f.f_max,
        normalize=f.normalize,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conformer-CTC.")
    parser.add_argument("--config", default="configs/conformer_ctc.yaml")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override config")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    set_seed(cfg.experiment.seed)

    tokenizer = load_tokenizer(args.tokenizer)
    fx = build_feature_extractor(cfg)

    spec_aug = (
        SpecAugment(
            cfg.spec_augment.freq_masks,
            cfg.spec_augment.freq_mask_width,
            cfg.spec_augment.time_masks,
            cfg.spec_augment.time_mask_ratio,
        )
        if cfg.spec_augment.enabled
        else None
    )

    train_ds = _AugmentingDataset(
        args.train_manifest, tokenizer, fx, cfg.tokenizer.lowercase, spec_augment=spec_aug
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_batch,
    )

    val_loader = None
    if args.val_manifest:
        val_ds = ASRDataset(args.val_manifest, tokenizer, fx, cfg.tokenizer.lowercase)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_batch,
        )

    model = ConformerCTC(cfg.model, tokenizer.vocab_size, tokenizer.blank_id)
    print(f"Model has {model.num_parameters() / 1e6:.1f}M parameters.")

    trainer = Trainer(model, tokenizer, cfg, train_loader, val_loader)
    trainer.fit()
    print(f"Done. Best WER: {trainer.best_wer:.3f}. Checkpoints in {cfg.experiment.output_dir}")


if __name__ == "__main__":
    main()
