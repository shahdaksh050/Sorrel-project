"""
Integrity-checked model persistence.

`pickle.load` executes code, so a model file must only be loaded if this
system wrote it. `save_model` writes the pickle plus an HMAC-SHA256 sidecar
(`<model>.sig`) keyed by a random secret kept next to the model in
`.model_key` (mode 0600, created on first save). `load_model` verifies the
sidecar before unpickling and raises `ModelIntegrityError` for a missing or
mismatching signature — so a pickle planted by anything other than
`save_model` (e.g. via a prompt-injected `model_path`) is never executed.

Threat model: this stops files the pipeline did not write. It does not stop an
attacker who can already read the key file, which is the same trust boundary
as the output directory itself.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import secrets
from pathlib import Path
from typing import Any

KEY_FILENAME = ".model_key"
SIG_SUFFIX = ".sig"


class ModelIntegrityError(Exception):
    """A model file is unsigned, or its signature does not match its bytes."""


def _key_for(model_path: Path, create: bool) -> bytes | None:
    key_path = model_path.parent / KEY_FILENAME
    if key_path.exists():
        return key_path.read_bytes()
    if not create:
        return None
    key = secrets.token_bytes(32)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


def _sig_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + SIG_SUFFIX)


def save_model(obj: Any, path: str | Path) -> Path:
    """Pickle `obj` to `path` and write its HMAC sidecar. Returns the path."""
    model_path = Path(path)
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
    """Verify the HMAC sidecar, then unpickle. Raises `ModelIntegrityError`."""
    model_path = Path(path)
    sig_path = _sig_path(model_path)
    key = _key_for(model_path, create=False)
    if key is None or not sig_path.exists():
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
