"""Testes de storage.py, que já está implementado."""

import json
from pathlib import Path

import numpy as np
import pytest

from juristcu import storage


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Direciona os artefatos para um diretório temporário.
    target = tmp_path / "artifacts"
    monkeypatch.setenv("JURISTCU_ARTIFACTS", str(target))
    return target


def test_artifacts_dir_uses_environment_variable(artifacts: Path) -> None:
    path = storage.artifacts_dir()
    assert path == artifacts.resolve()
    assert path.is_dir()


def test_embeddings_round_trip(artifacts: Path) -> None:
    x = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    path = storage.save_embeddings("sample", x)
    assert path == artifacts.resolve() / "embeddings" / "sample.npy"
    loaded = storage.load_embeddings("sample")
    assert loaded.dtype == x.dtype
    np.testing.assert_array_equal(loaded, x)


def test_load_missing_embeddings_raises(artifacts: Path) -> None:
    with pytest.raises(FileNotFoundError):
        storage.load_embeddings("missing")


def test_log_environment_has_expected_keys() -> None:
    environment = storage.log_environment()
    expected = {
        "python",
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "umap-learn",
        "nltk",
        "cuda",
        "gpu",
    }
    assert set(environment) == expected
    assert all(isinstance(value, str) for value in environment.values())
    assert environment["cuda"] in {"True", "False"}


def test_save_environment_writes_json(artifacts: Path) -> None:
    environment = storage.save_environment("01")
    path = artifacts.resolve() / "environment_01.json"
    assert json.loads(path.read_text(encoding="utf-8")) == environment
