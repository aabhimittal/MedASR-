"""Dataset and batching for CTC training.

The dataset is driven by a **manifest**: a JSON-lines file where each row is
one utterance::

    {"audio_filepath": "wavs/note_001.wav", "text": "patient denies chest pain"}
    {"audio_filepath": "wavs/note_002.wav", "duration": 4.1, "text": "..."}

This manifest format is deliberately the same one used by NeMo / ESPnet-style
toolkits, so existing clinical corpora can be dropped in unchanged.

Because utterances have different lengths, batching requires **padding** plus
explicit **length tensors**. CTC needs to know the true (unpadded) length of
both the acoustic sequence and the target text; padded positions must not
contribute to the loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from medasr.data.features import LogMelExtractor
from medasr.text import normalize
from medasr.tokenizer import BaseTokenizer


def _load_wav_stdlib(path: str) -> tuple[torch.Tensor, int]:
    """Read a PCM WAV with only the standard library (no third-party deps)."""
    import wave

    import numpy as np

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    data /= float(np.iinfo(dtype).max)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return torch.from_numpy(data.copy()), sr


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load a mono waveform, resampling to ``target_sr`` if needed.

    Resolution order: torchaudio (fast, many formats, can resample) ->
    soundfile -> the stdlib ``wave`` reader. The stdlib fallback means plain
    PCM WAVs load with no third-party audio dependency at all. Kept out of the
    hot path in tests, which construct tensors directly.
    """
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)  # downmix to mono
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav
    except ImportError:
        pass

    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)
    except ImportError:
        wav, sr = _load_wav_stdlib(path)

    if sr != target_sr:
        raise ValueError(
            f"{path} is {sr} Hz but target is {target_sr} Hz and torchaudio "
            "is not installed to resample. Install `medasr[audio]` or provide "
            f"{target_sr} Hz audio."
        )
    return wav


@dataclass
class Batch:
    """A padded, collated training batch.

    * ``features``       — ``(B, n_mels, T_max)`` padded log-mel features.
    * ``feature_lengths``— ``(B,)`` true frame count per utterance.
    * ``targets``        — ``(B, U_max)`` padded token ids.
    * ``target_lengths`` — ``(B,)`` true token count per utterance.
    """

    features: torch.Tensor
    feature_lengths: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    texts: Sequence[str]

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            features=self.features.to(device),
            feature_lengths=self.feature_lengths.to(device),
            targets=self.targets.to(device),
            target_lengths=self.target_lengths.to(device),
            texts=self.texts,
        )


class ASRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: BaseTokenizer,
        feature_extractor: LogMelExtractor,
        lowercase: bool = True,
        audio_loader: Optional[Callable[[str, int], torch.Tensor]] = None,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.lowercase = lowercase
        self.audio_loader = audio_loader or load_audio
        self.samples: List[dict] = self._read_manifest(manifest_path)

    @staticmethod
    def _read_manifest(path: str | Path) -> List[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"Manifest {path} is empty.")
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        sr = self.feature_extractor.sample_rate
        wav = self.audio_loader(row["audio_filepath"], sr)
        feats = self.feature_extractor(wav)  # (n_mels, T)

        text = normalize(row["text"], lowercase=self.lowercase)
        target = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return feats, target, text


def collate_batch(items) -> Batch:
    """Pad a list of ``(features, target, text)`` tuples into a :class:`Batch`."""
    feats_list, target_list, texts = zip(*items)

    feat_lengths = torch.tensor([f.shape[-1] for f in feats_list], dtype=torch.long)
    target_lengths = torch.tensor([t.numel() for t in target_list], dtype=torch.long)

    n_mels = feats_list[0].shape[0]
    t_max = int(feat_lengths.max().item())
    u_max = int(target_lengths.max().item()) if target_lengths.max() > 0 else 1

    features = torch.zeros(len(items), n_mels, t_max)
    targets = torch.zeros(len(items), u_max, dtype=torch.long)
    for i, (f, t) in enumerate(zip(feats_list, target_list)):
        features[i, :, : f.shape[-1]] = f
        targets[i, : t.numel()] = t

    return Batch(
        features=features,
        feature_lengths=feat_lengths,
        targets=targets,
        target_lengths=target_lengths,
        texts=list(texts),
    )
