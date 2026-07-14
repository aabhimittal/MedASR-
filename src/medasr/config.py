"""Typed configuration objects for MedASR.

We keep configuration as plain ``@dataclass`` objects rather than passing raw
dictionaries around the codebase. The benefits:

* **Autocomplete + type-checking** everywhere a config is used.
* **A single, documented schema** — the dataclass *is* the spec.
* **Validation on load** (e.g. the CTC input dim must match the mel channels).

A YAML file (see ``configs/conformer_ctc.yaml``) is parsed into these objects
via :meth:`Config.from_yaml`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    name: str = "conformer_ctc"
    seed: int = 1337
    output_dir: str = "runs/conformer_ctc"


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 0.0
    f_max: float = 8000.0
    normalize: bool = True


@dataclass
class SpecAugmentConfig:
    enabled: bool = True
    freq_masks: int = 2
    freq_mask_width: int = 27
    time_masks: int = 2
    time_mask_ratio: float = 0.05


@dataclass
class TokenizerConfig:
    type: str = "char"
    model_path: str = "artifacts/tokenizer.json"
    lowercase: bool = True


@dataclass
class ModelConfig:
    input_dim: int = 80
    d_model: int = 256
    num_layers: int = 16
    num_heads: int = 4
    ffn_expansion: int = 4
    conv_kernel_size: int = 31
    subsampling_factor: int = 4
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 100
    grad_accum_steps: int = 1
    max_grad_norm: float = 5.0
    optimizer: str = "adamw"
    lr: float = 1.5e-3
    weight_decay: float = 1e-6
    warmup_steps: int = 10000
    num_workers: int = 4
    device: str = "cuda"
    amp: bool = True
    log_every: int = 50
    eval_every: int = 1
    keep_last_n_checkpoints: int = 3


@dataclass
class DecodingConfig:
    strategy: str = "greedy"
    beam_size: int = 8


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a Config from a (possibly partial) nested dict."""
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            section = data.get(f.name, {}) or {}
            section_cls = f.type if is_dataclass(f.type) else _resolve(cls, f.name)
            kwargs[f.name] = _build_section(section_cls, section)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # -- validation ----------------------------------------------------------
    def validate(self) -> None:
        if self.model.input_dim != self.features.n_mels:
            raise ValueError(
                f"model.input_dim ({self.model.input_dim}) must equal "
                f"features.n_mels ({self.features.n_mels})."
            )
        if self.model.d_model % self.model.num_heads != 0:
            raise ValueError(
                f"model.d_model ({self.model.d_model}) must be divisible by "
                f"model.num_heads ({self.model.num_heads})."
            )
        if self.model.subsampling_factor != 4:
            raise ValueError("Only subsampling_factor=4 is currently supported.")
        if self.tokenizer.type not in {"char", "bpe"}:
            raise ValueError("tokenizer.type must be 'char' or 'bpe'.")
        if self.decoding.strategy not in {"greedy", "beam"}:
            raise ValueError("decoding.strategy must be 'greedy' or 'beam'.")


# -- helpers -----------------------------------------------------------------
_SECTION_TYPES = {
    "experiment": ExperimentConfig,
    "features": FeatureConfig,
    "spec_augment": SpecAugmentConfig,
    "tokenizer": TokenizerConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "decoding": DecodingConfig,
}


def _resolve(_cls, name: str):
    return _SECTION_TYPES[name]


def _build_section(section_cls, values: Dict[str, Any]):
    """Instantiate a section dataclass, ignoring unknown keys with a warning."""
    valid = {f.name for f in fields(section_cls)}
    unknown = set(values) - valid
    if unknown:
        import warnings

        warnings.warn(
            f"Ignoring unknown config keys for {section_cls.__name__}: {sorted(unknown)}",
            stacklevel=2,
        )
    return section_cls(**{k: v for k, v in values.items() if k in valid})
