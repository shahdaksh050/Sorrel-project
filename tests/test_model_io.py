"""Integrity-checked pickles: a model file the pipeline did not write is never executed."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.core.memory import MemorySystem
from src.core.model_io import ModelIntegrityError, load_model, save_model
from src.tools.base import ToolExecutionError
from src.tools.visualization import GenerateVisualizationsTool


class _Payload:
    """Unpickling calls `Path.write_text(marker, "x")` — a harmless stand-in for RCE."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (Path.write_text, (self.marker, "x"))


def test_round_trip(tmp_path: Path) -> None:
    path = save_model({"a": [1, 2, 3]}, tmp_path / "m.pkl")
    assert load_model(path) == {"a": [1, 2, 3]}


def test_unsigned_pickle_is_refused_and_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "PWNED"
    evil = tmp_path / "evil.pkl"
    evil.write_bytes(pickle.dumps(_Payload(marker)))
    with pytest.raises(ModelIntegrityError):
        load_model(evil)
    assert not marker.exists()


def test_tampered_pickle_is_refused_and_not_executed(tmp_path: Path) -> None:
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
    params = tool.prepare_params(
        {"chart_type": "feature_importance", "file_path": str(data),
         "target_column": "y", "model_path": str(evil)},
        MemorySystem(), str(out_root),
    )
    assert params["model_path"] != str(evil)
    with pytest.raises(ToolExecutionError):
        tool.execute(**params)
    assert not marker.exists()
