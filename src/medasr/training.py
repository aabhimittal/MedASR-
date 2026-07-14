"""Training loop, LR schedule, and checkpointing.

Transformer/Conformer models are famously sensitive to the learning-rate
schedule. The **Noam schedule** (Vaswani et al., 2017) linearly *warms up* the
LR for the first ``warmup_steps`` — while LayerNorm statistics and attention
are still random, a large LR diverges — then *decays* it proportionally to the
inverse square root of the step count::

    lr(step) = peak * min(step / warmup, (warmup / step) ** 0.5)

The :class:`Trainer` glues everything together: features -> model -> CTC loss
-> backward -> clip -> step, with gradient accumulation, optional mixed
precision on GPU, periodic validation (WER/CER), and rolling checkpoints.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from medasr.config import Config
from medasr.decoding import decode
from medasr.metrics import cer, wer
from medasr.models.ctc import ConformerCTC
from medasr.tokenizer import BaseTokenizer


class NoamWarmup:
    """Inverse-sqrt schedule with linear warmup, applied to an optimizer."""

    def __init__(self, optimizer: torch.optim.Optimizer, peak_lr: float, warmup_steps: int):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)
        self._step = 0

    def state_dict(self):
        return {"step": self._step}

    def load_state_dict(self, state):
        self._step = state["step"]

    def get_lr(self) -> float:
        step = max(1, self._step)
        scale = min(step / self.warmup_steps, (self.warmup_steps / step) ** 0.5)
        return self.peak_lr * scale

    def step(self) -> float:
        self._step += 1
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    if cfg.training.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


class Trainer:
    def __init__(
        self,
        model: ConformerCTC,
        tokenizer: BaseTokenizer,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = resolve_device(cfg.training.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = NoamWarmup(
            self.optimizer, cfg.training.lr, cfg.training.warmup_steps
        )
        self.use_amp = cfg.training.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.output_dir = Path(cfg.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.best_wer = math.inf

    # -- one epoch -----------------------------------------------------------
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        n = 0
        self.optimizer.zero_grad()
        start = time.time()

        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                log_probs, out_lengths = self.model(batch.features, batch.feature_lengths)
                loss = self.model.compute_loss(
                    log_probs, out_lengths, batch.targets, batch.target_lengths
                )
                loss = loss / self.cfg.training.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.cfg.training.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.max_grad_norm
                )
                lr = self.scheduler.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.cfg.training.log_every == 0:
                    speed = (i + 1) / (time.time() - start)
                    print(
                        f"epoch {epoch} step {self.global_step} "
                        f"loss {loss.item() * self.cfg.training.grad_accum_steps:.3f} "
                        f"lr {lr:.2e} ({speed:.1f} it/s)"
                    )

            running += loss.item() * self.cfg.training.grad_accum_steps
            n += 1
        return running / max(1, n)

    # -- validation ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.val_loader is None:
            return {}
        self.model.eval()
        refs, hyps = [], []
        for batch in self.val_loader:
            batch = batch.to(self.device)
            log_probs, out_lengths = self.model(batch.features, batch.feature_lengths)
            hyps.extend(
                decode(
                    log_probs,
                    out_lengths,
                    self.tokenizer,
                    self.cfg.decoding.strategy,
                    self.cfg.decoding.beam_size,
                )
            )
            refs.extend(batch.texts)
        return {"wer": wer(refs, hyps), "cer": cer(refs, hyps)}

    # -- full run ------------------------------------------------------------
    def fit(self) -> None:
        for epoch in range(1, self.cfg.training.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            msg = f"[epoch {epoch}] train_loss={train_loss:.3f}"
            if epoch % self.cfg.training.eval_every == 0 and self.val_loader is not None:
                metrics = self.evaluate()
                msg += f" val_wer={metrics['wer']:.3f} val_cer={metrics['cer']:.3f}"
                if metrics["wer"] < self.best_wer:
                    self.best_wer = metrics["wer"]
                    self.save_checkpoint("best.pt", epoch)
            print(msg)
            self.save_checkpoint(f"epoch_{epoch}.pt", epoch)
            self._prune_checkpoints()

    # -- checkpointing -------------------------------------------------------
    def save_checkpoint(self, name: str, epoch: int) -> Path:
        path = self.output_dir / name
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": self.cfg.to_dict(),
                "best_wer": self.best_wer,
            },
            path,
        )
        return path

    def _prune_checkpoints(self) -> None:
        keep = self.cfg.training.keep_last_n_checkpoints
        ckpts = sorted(
            self.output_dir.glob("epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for p in ckpts[:-keep]:
            p.unlink(missing_ok=True)
