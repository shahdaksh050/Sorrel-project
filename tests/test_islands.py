"""Tests for the optional UI-island plumbing (availability, URLs, no HTML injection, contrast)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from ui.components.islands import ISLANDS_DIR, island_base_url, islands_available

from src.core.design_tokens import palette

ROOT = Path(__file__).resolve().parents[1]


def _write_build(root: Path, *, with_file: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({"src/hero/main.tsx": {"file": "hero.js", "name": "hero", "isEntry": True}}),
        encoding="utf-8",
    )
    if with_file:
        (root / "hero.js").write_text("export {};", encoding="utf-8")


def test_available_with_manifest_and_file(tmp_path: Path) -> None:
    _write_build(tmp_path)
    assert islands_available("hero", tmp_path)
    assert not islands_available("progress", tmp_path)


def test_unavailable_when_dir_missing(tmp_path: Path) -> None:
    assert not islands_available("hero", tmp_path / "nope")
    assert island_base_url("hero", tmp_path / "nope") == ""


def test_unavailable_when_listed_file_missing(tmp_path: Path) -> None:
    _write_build(tmp_path, with_file=False)
    assert not islands_available("hero", tmp_path)


@pytest.mark.parametrize("content", ["not json", "[]", "{\"a\": 3}"])
def test_corrupt_manifest_is_unavailable(tmp_path: Path, content: str) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "manifest.json").write_text(content, encoding="utf-8")
    assert not islands_available("hero", tmp_path)


def test_base_url_honours_base_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import streamlit as st

    _write_build(tmp_path)
    monkeypatch.setattr(st, "get_option", lambda k: "/dsa" if k == "server.baseUrlPath" else None)
    assert island_base_url("hero", tmp_path) == "/dsa/app/static/islands/"
    monkeypatch.setattr(st, "get_option", lambda k: "")
    assert island_base_url("hero", tmp_path) == "/app/static/islands/"


def test_committed_build_is_usable() -> None:
    assert islands_available("hero", ISLANDS_DIR)


def test_island_sources_never_inject_html() -> None:
    src = ROOT / "ui" / "islands" / "src"
    for f in src.rglob("*"):
        if f.suffix in {".ts", ".tsx", ".js", ".jsx"}:
            text = f.read_text(encoding="utf-8")
            assert "dangerouslySetInnerHTML" not in text, f
            assert not re.search(r"\.innerHTML\s*=", text), f


def _lum(hex_: str) -> float:
    rgb = [int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _mix(fg: str, bg: str, alpha: float) -> str:
    a = [int(fg[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * alpha + y * (1 - alpha)):02x}" for x, y in zip(a, b, strict=False))


def _ratio(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("theme", ["day", "night"])
def test_spotlight_peak_tint_keeps_text_aa(theme: str) -> None:
    """Hero text must stay >=4.5:1 over the strongest spotlight tint (11% --accent)."""
    p = palette(theme)
    css = (ROOT / "ui" / "islands" / "src" / "hero" / "hero.css").read_text(encoding="utf-8")
    peak = max(int(m) for m in re.findall(r"var\(--accent\) (\d+)%", css)) / 100
    tinted = _mix(p["accent"], p["stock"], peak)
    assert _ratio(p["ink"], tinted) >= 4.5
    assert _ratio(p["graphite"], tinted) >= 4.5
    # Accent-coloured hero text uses the accent-as-text role: raw Night `pen` is a fill (about 2.5:1 as text).
    assert _ratio(p["accent_text"], tinted) >= 4.5
