"""Integrity-checked model persistence.

Pickle deserialisation can execute code, so :func:`load_model` only unpickles
bytes that carry a valid HMAC sidecar made by this installation.  The signing
key is deliberately kept in an operator-owned configuration directory, never
beside a model, run artifact, or export.  Configure its exact location with
``MODEL_SIGNING_KEY_PATH`` or its parent directory with
``DSA_MODEL_KEY_DIR``.  On POSIX, newly-created key material is private to the
current user (directories ``0700``; key ``0600``).

There is intentionally no migration path that accepts old unsigned pickles:
they remain unsafe and are refused before ``pickle.loads`` is reached.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import secrets
from pathlib import Path
from typing import Any

KEY_FILENAME = "model-signing.key"
SIG_SUFFIX = ".sig"


class ModelIntegrityError(Exception):
    """A model is unsigned, unverifiable, or has unsafe key configuration."""


def _configured_key_path() -> Path:
    """Return the operator-controlled key path without creating it."""
    explicit = os.getenv("MODEL_SIGNING_KEY_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    key_dir = os.getenv("DSA_MODEL_KEY_DIR")
    base = Path(key_dir).expanduser() if key_dir else Path.home() / ".dsa_agent"
    return (base / KEY_FILENAME).resolve()


def _key_is_outside_artifact_roots(key_path: Path, model_path: Path) -> bool:
    """Reject keys below a model directory or named run/output/export root."""
    artifact_root_names = {"model", "models", "output", "run", "runs", "export", "exports"}
    roots = [parent for parent in model_path.parents if parent.name.lower() in artifact_root_names]
    for root in roots:
        try:
            key_path.relative_to(root)
        except ValueError:
            continue
        return False
    return True


def _private_mode(path: Path, expected: int) -> bool:
    """Check POSIX permissions; Windows ACLs are managed by the OS instead."""
    if os.name == "nt":
        return True
    return path.stat().st_mode & 0o777 == expected


def _key_for(model_path: Path, create: bool) -> bytes | None:
    key_path = _configured_key_path()
    if not _key_is_outside_artifact_roots(key_path, model_path):
        raise ModelIntegrityError("MODEL_SIGNING_KEY_PATH must be outside model/run/export roots.")
    if key_path.parent.exists() and not _private_mode(key_path.parent, 0o700):
        raise ModelIntegrityError("Model signing key directory has unsafe permissions.")
    if key_path.exists():
        if not key_path.is_file() or not _private_mode(key_path, 0o600):
            raise ModelIntegrityError("Model signing key has unsafe type or permissions.")
        key = key_path.read_bytes()
        if len(key) < 32:
            raise ModelIntegrityError("Model signing key is invalid.")
        return key
    if not create:
        return None
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not _private_mode(key_path.parent, 0o700):
        raise ModelIntegrityError("Model signing key directory has unsafe permissions.")
    key = secrets.token_bytes(32)
    try:
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _key_for(model_path, create=False)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    return key


def _sig_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + SIG_SUFFIX)


def save_model(obj: Any, path: str | Path) -> Path:
    """Pickle ``obj`` to ``path`` and write a verifiable HMAC sidecar."""
    model_path = Path(path).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(obj)
    key = _key_for(model_path, create=True)
    assert key is not None
    model_path.write_bytes(data)
    _sig_path(model_path).write_text(
        hmac.new(key, data, hashlib.sha256).hexdigest(), encoding="utf-8"
    )
    return model_path


def load_model(path: str | Path) -> Any:
    """Verify the HMAC sidecar, then unpickle; never accept unsigned models."""
    model_path = Path(path).resolve()
    sig_path = _sig_path(model_path)
    key = _key_for(model_path, create=False)
    if key is None or not sig_path.is_file() or not model_path.is_file():
        raise ModelIntegrityError(
            f"{model_path.name} has no integrity signature; refusing to load an unverified pickle."
        )
    data = model_path.read_bytes()
    expected = hmac.new(key, data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig_path.read_text(encoding="utf-8").strip()):
        raise ModelIntegrityError(
            f"{model_path.name} does not match its integrity signature; refusing to load it."
        )
    return pickle.loads(data)
