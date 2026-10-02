"""
Coercing dataset reader shared by the controller and every tool.

Lives in `core` so the controller can read a dataset the way tools do without
importing from `src/tools/` (AGENTS.md layer rules). Raises `DatasetReadError`;
`src.tools.data_processing._read_df` wraps it into `ToolExecutionError`.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import pandas as pd

from src.core.coercion import coerce_types
from src.core.io import _READ_CACHE, _READ_CACHE_LOCK, _cache_key, read_any

#: Coerced frames by the read cache's (path, mtime, size) key. Coercing a
#: 500k-row export costs seconds per call and every tool re-reads, so the
#: repaired frame is kept for the few most recent files. An entry is only
#: served while the raw frame is still in src.core.io's cache: a writer's
#: invalidate_read_cache() (the guard against same-tick rewrites) drops both.
_COERCED_MAX_ENTRIES = 2
_COERCED: OrderedDict[tuple[str, int, int], pd.DataFrame] = OrderedDict()
_COERCED_LOCK = threading.Lock()
_COERCED_KEY_LOCKS: dict[tuple[str, int, int], threading.Lock] = {}


def read_analysis_df(file_path: str) -> pd.DataFrame:
    """
    Read a dataset ready for analysis: unified reader + type coercion.

    Coercion belongs here, not at individual call sites. A retail export
    carries money as "$1,234.56" and rates as "45.3%", which read back as
    strings. `controller.load_dataset` coerces before profiling, so the
    *profile* saw them as numeric — but every tool re-read the file through
    this helper and got the strings back, so revenue was invisible to the
    entire analysis. On a real sales file that left correlation running on
    a customer ID and a quantity, and reporting r=-0.06 between them as the
    headline finding, while never once looking at revenue.

    Coercion is idempotent, and every repair is recorded and reported by
    the ingestion path (memory context "coercions" -> the report's Data
    Overview), so nothing here is silent.
    """
    key = _cache_key(file_path)
    if key is None:
        return coerce_types(read_any(file_path)[0])[0]
    # One lock per file: tools run in parallel batches, and N threads coercing
    # the same 500k rows at once would each pay N times the GIL-bound cost.
    with _COERCED_LOCK:
        key_lock = _COERCED_KEY_LOCKS.setdefault(key, threading.Lock())
    with key_lock:
        with _READ_CACHE_LOCK:
            raw_cached = key in _READ_CACHE
        with _COERCED_LOCK:
            hit = _COERCED.get(key) if raw_cached else _COERCED.pop(key, None)
            if hit is not None:
                _COERCED.move_to_end(key)
        if hit is not None:
            return hit.copy()
        df = read_any(file_path)[0]
        repaired, _coercions = coerce_types(df)
        with _COERCED_LOCK:
            _COERCED[key] = repaired.copy()
            while len(_COERCED) > _COERCED_MAX_ENTRIES:
                _COERCED_KEY_LOCKS.pop(_COERCED.popitem(last=False)[0], None)
        return repaired
