"""
Unit tests for Deep Nested JSON Ingestion (Phase 11).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.core.io import read_any


def test_deep_nested_json_unwrapping(tmp_path: Path) -> None:
    json_path = tmp_path / "nested_api_response.json"
    payload = {
        "status": "success",
        "code": 200,
        "data": [
            {
                "user": {"id": 1, "profile": {"name": "Alice", "age": 30}},
                "metrics": {"login_count": 12, "active": True},
            },
            {
                "user": {"id": 2, "profile": {"name": "Bob", "age": 25}},
                "metrics": {"login_count": 4, "active": False},
            },
        ],
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    df, report = read_any(str(json_path))

    assert len(df) == 2
    assert "user.id" in df.columns or "user.profile.name" in df.columns
    assert "metrics.login_count" in df.columns
    assert report.flattened is True
    assert any("Unpacked nested records list" in n for n in report.notes)
