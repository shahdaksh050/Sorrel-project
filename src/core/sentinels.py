"""
Numeric placeholder ("sentinel") detection — -200, -999, 9999 used as "missing".

coercion.py only recognises sentinels that are *strings* ("N/A", "?"). A
number typed into a numeric column as a placeholder (UCI Air Quality's -200
across a dozen sensors) sails through as data: charts spike at -200, axes
stretch, means and correlations are wrong, and nothing says so.

The detector is data-driven and deliberately conservative — a false positive
erases real data, a miss only leaves a visible spike. A value is nulled only
when it is an exact repeat, the column min or max, far outside the rest of
the column, and not one of the values that are routinely real (0, 1, 100).
Repetition of the same far-out value across several columns is treated as
strong evidence and relaxes the count threshold.

Pure function of a DataFrame; no I/O. Every null is returned as a record so
callers disclose it (degradations log) instead of hiding it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MIN_ROWS = 30
MIN_COUNT = 5
MIN_SHARE = 0.01
#: Cross-column recurrence relaxes the count floor to this.
RELAXED_COUNT = 3
RELAXED_SHARE = 0.001
#: Columns (including the column itself) that must show the same far-out value
#: for the relaxed count — and the higher share cap — to apply.
RECURRENCE_COLUMNS = 3
MAX_SHARE = 0.40
MAX_SHARE_RECURRING = 0.95
#: "Atom at the extreme": the value is a heavy spike at the column min/max,
#: this many times more frequent than any of its nearest inside neighbours.
ATOM_DOMINANCE = 10
ATOM_NEIGHBOURS = 5
LOW_CARDINALITY = 8
MIN_REST_DISTINCT = 10
FENCE_K = 3.0
GAP_K_RECOGNISED = 1.5
GAP_K_OTHER = 3.0

#: Placeholder codes people actually use — a lower bar for the gap test only.
RECOGNISED_CODES: frozenset[float] = frozenset(
    {-9999.0, -999.0, -99.0, -1.0, -200.0, 9999.0, 99999.0, 999.0}
)
#: Routinely legitimate extremes (zero salary, 0/1 flags, 100% caps).
NEVER_SENTINEL: frozenset[float] = frozenset({0.0, 1.0, 100.0})


def format_value(v: float) -> str:
    return f"{v:.0f}" if v == int(v) else f"{v:g}"


def sentinel_note(column: str, value: float, count: int, share: float) -> str:
    return (
        f"{column}: {count:,} values of {format_value(value)} treated as missing "
        f"({share:.1%} of non-null values; a repeated placeholder far outside the rest of the column)."
    )


def _eligible(name: str, series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return False
    from src.core.profiler import has_identifier_name_hint

    return not has_identifier_name_hint(name)


def _extreme_atom(
    rest: np.ndarray, v: float, side: int, count: int, n: int, scale: float
) -> str | None:
    """Classify a spike at the column edge against its inside neighbours.

    Catches placeholders in wide-spread sensor columns where the 3*IQR fence
    test is too strict. Returns None (not an atom), "strong" (also clears the
    inside neighbour by at least one IQR: enough on its own) or "weak"
    (dominant, but the gap is under one IQR: only trusted when the same value
    is an atom in RECURRENCE_COLUMNS columns). A neighbour of comparable
    frequency, or a tiny gap relative to the IQR in zero-inflated / capped
    data, never yields "strong".
    """
    if count < max(MIN_COUNT, MIN_SHARE * n):
        return None
    vals, counts = np.unique(rest, return_counts=True)
    near = slice(0, ATOM_NEIGHBOURS) if side < 0 else slice(-ATOM_NEIGHBOURS, None)
    if count < ATOM_DOMINANCE * int(counts[near].max()):
        return None
    inside = float(vals[0] if side < 0 else vals[-1])
    return "strong" if abs(v - inside) >= scale else "weak"


def _candidates(x: np.ndarray) -> list[tuple[float, int, bool]]:
    """(value, count, weak) for the column's min/max when it looks like a
    placeholder, tested with the relaxed count floor; the caller applies the
    strict one. `weak` marks an atom that needs cross-column recurrence."""
    n = x.size
    if n < MIN_ROWS:
        return []
    lo, hi = float(x.min()), float(x.max())
    if lo == hi or pd.unique(x).size <= LOW_CARDINALITY:
        return []
    found: list[tuple[float, int, bool]] = []
    pool = x
    for v, side in ((lo, -1), (hi, 1)):
        if v in NEVER_SENTINEL:
            continue
        mask = pool == v
        count = int(mask.sum())
        if count < max(RELAXED_COUNT, RELAXED_SHARE * n) or count / n >= MAX_SHARE_RECURRING:
            continue
        rest = pool[~mask]
        if pd.unique(rest).size < MIN_REST_DISTINCT:
            continue
        q1, q3 = np.percentile(rest, [25, 75])
        scale = float(q3 - q1)
        if scale <= 0:
            scale = float(rest.std())
        if not scale > 0:
            continue
        beyond = (q1 - v) if side < 0 else (v - q3)
        gap = abs(v - float(rest.min() if side < 0 else rest.max()))
        k = GAP_K_RECOGNISED if v in RECOGNISED_CODES else GAP_K_OTHER
        if beyond > FENCE_K * scale and gap > k * scale:
            found.append((v, count, False))
            pool = rest
            continue
        atom = _extreme_atom(rest, v, side, count, n, scale)
        if atom is not None:
            found.append((v, count, atom == "weak"))
            pool = rest
    return found


def null_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Null placeholder values in numeric columns.

    Returns (df, records); df is the input object when nothing was found.
    Each record: column, value, count, share (of non-null values), note.
    """
    per_column: dict[int, tuple[int, list[tuple[float, int, bool]]]] = {}
    recurrence: dict[float, int] = {}
    for i, col in enumerate(df.columns):
        series = df.iloc[:, i]
        if not _eligible(str(col), series):
            continue
        x = series.dropna().to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        cands = _candidates(x)
        if not cands:
            continue
        per_column[i] = (x.size, cands)
        for v, _, _w in cands:
            recurrence[v] = recurrence.get(v, 0) + 1

    records: list[dict[str, Any]] = []
    out: pd.DataFrame | None = None
    for i, (n, cands) in per_column.items():
        series = df.iloc[:, i]
        drop = [
            (v, c) for v, c, weak in cands
            if (not weak and c >= max(MIN_COUNT, MIN_SHARE * n) and c / n < MAX_SHARE)
            or recurrence.get(v, 0) >= RECURRENCE_COLUMNS
        ]
        for v, c in drop:
            series = series.mask((series == v).fillna(False).astype(bool))
            records.append({
                "column": str(df.columns[i]),
                "value": v,
                "count": c,
                "share": round(c / n, 4),
                "note": sentinel_note(str(df.columns[i]), v, c, c / n),
            })
        if drop:
            if out is None:
                out = df.copy(deep=False)
            out.isetitem(i, series)
    return (out if out is not None else df), records
