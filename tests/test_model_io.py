"""Integrity-checked pickles: a model file the pipeline did not write is never executed."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.core.memory import DatasetMetadata, MemorySystem
from src.core.model_io import ModelIntegrityError, load_model, save_model
from src.tools.base import ToolExecutionError
from src.tools.visualization import GenerateVisualizationsTool


class _Payload:
    """Unpickling calls `Path.write_text(marker, "x")` — a harmless stand-in for RCE."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (Path.write_text, (self.marker, "x"))


def test_round_trip_uses_an_operator_key_outside_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_dir = tmp_path / "operator-config"
    monkeypatch.setenv("DSA_MODEL_KEY_DIR", str(key_dir))
    path = save_model({"a": [1, 2, 3]}, tmp_path / "m.pkl")
    assert load_model(path) == {"a": [1, 2, 3]}
    assert (key_dir / "model-signing.key").is_file()
    assert not (tmp_path / ".model_key").exists()


def test_unsigned_pickle_is_refused_and_not_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DSA_MODEL_KEY_DIR", str(tmp_path / "operator-config"))
    marker = tmp_path / "PWNED"
    evil = tmp_path / "evil.pkl"
    evil.write_bytes(pickle.dumps(_Payload(marker)))
    with pytest.raises(ModelIntegrityError):
        load_model(evil)
    assert not marker.exists()


def test_tampered_pickle_is_refused_and_not_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DSA_MODEL_KEY_DIR", str(tmp_path / "operator-config"))
    marker = tmp_path / "PWNED"
    path = save_model({"ok": 1}, tmp_path / "m.pkl")
    path.write_bytes(pickle.dumps(_Payload(marker)))  # swap bytes, keep old signature
    with pytest.raises(ModelIntegrityError):
        load_model(path)
    assert not marker.exists()


def test_visualization_pins_model_path_and_never_unpickles_foreign_file(tmp_path: Path) -> None:
    marker = tmp_path / "PWNED"
    evil = tmp_path / "evil.pkl"  # outside the run's output root
    evil.write_bytes(pickle.dumps(_Payload(marker)))
    data = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2, 3, 4], "y": [1, 0, 1, 0]}).to_csv(data, index=False)
    out_root = tmp_path / "out"
    out_root.mkdir()

    tool = GenerateVisualizationsTool()
    memory = MemorySystem()
    memory.dataset_metadata = DatasetMetadata(
        file_path=str(data),
        row_count=4,
        column_count=2,
        columns={"a": "int64", "y": "int64"},
        missing_values={"a": 0, "y": 0},
        numerical_cols=["a", "y"],
        categorical_cols=[],
    )
    params = tool.prepare_params(
        {"chart_type": "feature_importance", "file_path": str(data),
         "target_column": "y", "model_path": str(evil)},
        memory, str(out_root),
    )
    assert params["model_path"] != str(evil)
    with pytest.raises(ToolExecutionError):
        tool.execute(**params)
    assert not marker.exists()


def test_key_path_inside_model_artifacts_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "models" / "safe.pkl"
    monkeypatch.setenv("MODEL_SIGNING_KEY_PATH", str(model.parent / "model-signing.key"))
    with pytest.raises(ModelIntegrityError, match="outside model/run/export roots"):
        save_model({"safe": True}, model)


@pytest.mark.skipif(os.name == "nt", reason="Windows uses ACLs, not POSIX mode bits")
def test_overly_permissive_existing_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = tmp_path / "operator-config" / "model-signing.key"
    key.parent.mkdir()
    key.write_bytes(b"x" * 32)
    os.chmod(key, 0o644)
    monkeypatch.setenv("MODEL_SIGNING_KEY_PATH", str(key))
    with pytest.raises(ModelIntegrityError, match="permissions"):
        save_model({"safe": True}, tmp_path / "models" / "safe.pkl")
