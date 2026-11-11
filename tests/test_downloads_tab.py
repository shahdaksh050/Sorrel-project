"""Artifact-download filtering is fail-closed for secret-bearing model roots."""
from __future__ import annotations

from pathlib import Path

from ui.tabs.downloads_tab import _downloadable_models


def test_downloadable_models_only_returns_regular_pickles(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    allowed = models / "trained.pkl"
    allowed.write_bytes(b"model")
    (models / "trained.pkl.sig").write_text("signature", encoding="utf-8")
    (models / ".model_key").write_text("secret", encoding="utf-8")
    (models / "model-signing.key").write_text("secret", encoding="utf-8")
    (models / "notes.txt").write_text("not a model", encoding="utf-8")

    assert _downloadable_models(models) == [allowed]


def test_downloadable_models_rejects_a_symlinked_directory(tmp_path: Path) -> None:
    models = tmp_path / "models"
    target = tmp_path / "target"
    target.mkdir()
    (target / "trained.pkl").write_bytes(b"model")
    try:
        models.symlink_to(target, target_is_directory=True)
    except OSError:
        # Symlink creation is often unavailable on Windows CI; the non-link
        # filtering above is still covered there.
        return

    assert _downloadable_models(models) == []
