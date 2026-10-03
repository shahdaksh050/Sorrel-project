"""Availability + URLs for the optional display-only UI islands (built from ``ui/islands``).

The built bundle is committed under ``static/islands/``. Nothing here requires Node: when the
bundle is missing or corrupt the islands are simply reported unavailable and the page renders
its plain HTML/CSS.
"""
from __future__ import annotations

import json
from pathlib import Path

ISLANDS_DIR: Path = Path(__file__).resolve().parents[2] / "static" / "islands"


def islands_available(name: str, islands_dir: Path | None = None) -> bool:
    """True when ``name`` is in the build manifest and every file it needs exists."""
    root = islands_dir or ISLANDS_DIR
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        entry = next(v for v in manifest.values() if isinstance(v, dict) and v.get("name") == name)
        files = [entry["file"], *entry.get("imports_files", [])]
        return bool(files) and all((root / str(f)).is_file() for f in files)
    except (OSError, ValueError, StopIteration, KeyError, AttributeError):
        return False


def island_base_url(name: str, islands_dir: Path | None = None) -> str:
    """Base URL (trailing slash) of the islands folder, or ``""`` when ``name`` is unavailable."""
    if not islands_available(name, islands_dir):
        return ""
    from ui.styles import static_url

    return static_url("islands/")
