from pathlib import Path

import pytest

from medasr.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_defaults_are_consistent():
    cfg = Config()
    cfg.validate()  # should not raise
    assert cfg.model.input_dim == cfg.features.n_mels


def test_from_dict_partial_override():
    cfg = Config.from_dict({"model": {"d_model": 128, "num_heads": 4}})
    assert cfg.model.d_model == 128
    assert cfg.model.num_layers == 16  # default preserved


def test_validation_rejects_dim_mismatch():
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"input_dim": 40}, "features": {"n_mels": 80}})


def test_validation_rejects_bad_head_divisor():
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"d_model": 100, "num_heads": 3}})


def test_yaml_config_loads(tmp_path):
    cfg = Config()
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    reloaded = Config.from_yaml(path)
    assert reloaded.model.d_model == cfg.model.d_model


def test_unknown_keys_warn(recwarn):
    Config.from_dict({"model": {"nonsense": 1}})
    assert any("Ignoring unknown" in str(w.message) for w in recwarn.list)


def test_repo_config_is_valid():
    Config.from_yaml(REPO_ROOT / "configs" / "conformer_ctc.yaml").validate()
