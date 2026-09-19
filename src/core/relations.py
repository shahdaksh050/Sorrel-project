"""
Multi-column relation discovery — which numeric columns are exact or
near-exact formulas of each other.

`revenue = price * quantity`, `profit = revenue - cost`, `total = a + b + c`,
`cum_sales` running-total columns. Knowing this matters twice over: the
planner can build derived metrics from the identity instead of re-deriving it
badly, and it must not treat a formula column and its inputs as independent
drivers (correlation ranking) or as features for a model that predicts one of
them (leakage).

Pure computation — no I/O, no LLM, deterministic. Cost is bounded by design:
at most `MAX_CANDIDATES` numeric columns are ever paired, screening runs on an
evenly-spaced row sample with vectorised array maths (pairs and triples only,
never wider enumeration), and only the few survivors are confirmed on the full
columns. Relations are tested in one orientation only (a product `a = b*c`,
a sum `a = b+c[+d]`); the algebraically equivalent forms (`b = a/c`,
`b = a-c`) are the same relation and are reported once, oriented for reading
by `_orient`.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

#: Numeric columns considered at once — pairing cost grows with this cubed.
MAX_CANDIDATES = 15

#: Fewer complete rows than this and no relation is credible.
MIN_ROWS = 30

#: Relations reported per dataset (best first).
MAX_RELATIONS = 8

_SCREEN_TOL = 0.05        # per-row relative error tolerated while screening
_SCREEN_SHARE = 0.9       # share of sample rows that must fit within it
_NEAR_R2 = 0.999          # full-column R^2 floor (exact and near)
_NEAR_P95 = 0.05          # near: 95th-percentile relative error ceiling
_EXACT_ABS = 1e-9         # exact: float noise, as a share of the target's max
_TRIPLE_ROWS = 400        # rows for the three-term screen (a 4-D array)
_PER_TARGET = 6           # screening survivors confirmed per target
_FULL_ROWS_CAP = 2_000_000
_SEQ_ROWS = 50_000        # contiguous prefix for cumulative/lag identities
_MAX_PARTS = 8            # widest "total = sum of parts" the NNLS fallback reports

#: Name tokens that mark the natural "result" of a subtraction / a quotient,
#: used only to pick a readable orientation for an already-confirmed relation.
_DIFF_TOKENS = frozenset({
    "profit", "net", "diff", "difference", "delta", "change", "gap",
    "balance", "remaining", "variance", "surplus", "deficit",
})
_RATE_TOKENS = frozenset({
    "rate", "ratio", "pct", "percent", "percentage", "share", "margin",
    "avg", "average", "mean", "price", "per", "unit",
})


def _tokens(name: str) -> set[str]:
    return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()))


def _fmt(name: str) -> str:
    """Column name as it appears in an expression (`backticked` when it is
    not a plain identifier — dsa.derive reads the same syntax)."""
    return name if name.isidentifier() else f"`{name}`"


def _floats(series: pd.Series[Any]) -> np.ndarray:
    values: np.ndarray = series.to_numpy(dtype="float64", na_value=np.nan)
    values[~np.isfinite(values)] = np.nan
    return values


def _is_progression(head: np.ndarray) -> bool:
    """A row-index-like column (constant non-zero step) is never a formula."""
    if len(head) < 3 or not np.isfinite(head).all():
        return False
    step = np.diff(head)
    return bool(step[0] != 0 and np.allclose(step, step[0], rtol=0, atol=1e-9 * max(1.0, abs(step[0]))))


def _candidates(
    df: pd.DataFrame, idx: np.ndarray, skip: set[str]
) -> tuple[list[str], list[int], np.ndarray]:
    """Rank numeric columns and return (names, positions, sample matrix) for
    the best `MAX_CANDIDATES`: booleans, constants, binary flags, row-index
    progressions, mostly-null and `skip`ped (id/PII/dimension) columns drop out."""
    dupes = {k for k, v in Counter(str(c) for c in df.columns).items() if v > 1}
    scored: list[tuple[tuple[float, float, str], int, str, np.ndarray]] = []
    for pos, raw_name in enumerate(df.columns):
        name = str(raw_name)
        series = df.iloc[:, pos]
        if name in skip or name in dupes:
            continue
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        sample = _floats(series.iloc[idx])
        finite = sample[np.isfinite(sample)]
        if len(finite) < MIN_ROWS or len(finite) < 0.5 * len(sample):
            continue
        if finite.min() == finite.max() or np.unique(finite).size <= 2:
            continue
        if _is_progression(_floats(series.iloc[:2000])):
            continue
        cv = float(finite.std() / (abs(finite.mean()) + 1e-12))
        scored.append(((-round(len(finite) / len(sample), 3), -cv, name), pos, name, sample))
    scored.sort(key=lambda item: item[0])
    top = sorted(scored[:MAX_CANDIDATES], key=lambda item: item[1])
    if not top:
        return [], [], np.empty((len(idx), 0))
    return [t[2] for t in top], [t[1] for t in top], np.column_stack([t[3] for t in top])


def _share(target: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Share of valid sample rows where `fit` (rows on axis 0, any number of
    candidate axes after it) matches `target` within the screening tolerance."""
    shape = (-1,) + (1,) * (fit.ndim - 1)
    with np.errstate(all="ignore"):
        seen = np.abs(target[np.isfinite(target)])
        floor = max(0.01 * float(np.median(seen)) if seen.size else 1.0, 1e-12)
        rel = np.abs(fit - target.reshape(shape)) / np.maximum(np.abs(target), floor).reshape(shape)
        valid = np.isfinite(rel)
        n_valid = valid.sum(axis=0)
        hits = (rel <= _SCREEN_TOL).sum(axis=0)
        share = hits / np.maximum(n_valid, 1)
    return np.where(n_valid >= MIN_ROWS, share, 0.0)


def _nnls_parts(mat: np.ndarray, i: int) -> tuple[int, ...] | None:
    """Sparse unit-coefficient decomposition of column `i` into >=3 other
    columns (a total with four or more parts is beyond the triple screen)."""
    from scipy.optimize import nnls

    others = [j for j in range(mat.shape[1]) if j != i]
    rows = np.isfinite(mat).all(axis=1)
    if rows.sum() < MIN_ROWS or len(others) < 3 or (mat[rows, i] < 0).any():
        return None
    try:
        coef, _ = nnls(mat[np.ix_(rows, others)], mat[rows, i])
    except (RuntimeError, ValueError):
        return None
    used = [(others[j], c) for j, c in enumerate(coef) if c > 1e-6]
    if not 3 <= len(used) <= _MAX_PARTS or any(abs(c - 1.0) > 0.02 for _, c in used):
        return None
    return tuple(j for j, _ in used)


def _combine(family: str, parts: list[np.ndarray]) -> np.ndarray:
    combined: np.ndarray
    with np.errstate(all="ignore"):
        combined = np.prod(parts, axis=0) if family == "mul" else np.sum(parts, axis=0)
    return combined


def _decimals(values: np.ndarray, scale: float) -> int | None:
    """Fewest decimals (0-6) the values are rounded to, or None."""
    for d in range(7):
        if float(np.max(np.abs(values - np.round(values, d)))) <= _EXACT_ABS * max(1.0, scale):
            return d
    return None


def _confirm(family: str, y: np.ndarray, parts: list[np.ndarray]) -> dict[str, Any] | None:
    """Test `y = combine(parts)` on the full columns (rows where every column
    is present). Returns the fit statistics, or None if it is neither exact
    nor near-exact, or if any single term is not actually needed."""
    ok = np.isfinite(y)
    n_target = int(ok.sum())
    for p in parts:
        ok &= np.isfinite(p)
    n = int(ok.sum())
    if n < MIN_ROWS or n < 0.5 * n_target:
        return None
    tv, pv = y[ok], [p[ok] for p in parts]
    fit = _combine(family, pv)
    err = np.abs(tv - fit)
    scale = float(np.abs(tv).max())
    sst = float(((tv - tv.mean()) ** 2).sum())
    if not np.isfinite(err).all() or sst <= 0 or scale == 0:
        return None
    sse = float((err**2).sum())
    r2 = 1.0 - sse / sst
    if r2 < _NEAR_R2:
        return None
    rel = err / np.maximum(np.abs(tv), max(0.01 * float(np.median(np.abs(tv))), 1e-12 * scale))
    worst = float(err.max())
    dec = None if worst <= _EXACT_ABS * scale else _decimals(tv, scale)
    exact = worst <= _EXACT_ABS * scale + (0.0 if dec is None else 0.5 * 10.0**-dec * (1 + 1e-9))
    if not exact and float(np.quantile(rel, 0.95)) > _NEAR_P95:
        return None
    for j in range(len(pv)):
        alt = _combine(family, pv[:j] + pv[j + 1:])
        if float(((tv - alt) ** 2).sum()) <= max(4.0 * sse, (1e-6 * scale) ** 2 * n):
            return None  # this term adds nothing — the rest already explain y
    return {
        "r2": round(r2, 6),
        "max_rel_err": float(f"{float(rel.max()):.3g}"),
        "exact": bool(exact),
        "n_rows": n,
    }


def _orient(
    family: str, target: str, terms: list[str], med: dict[str, float]
) -> tuple[str, str, list[str], str]:
    """Readable orientation of a confirmed relation -> (kind, target, terms, expr).
    `revenue = price * qty` may read better as `price = revenue / qty`, and
    `total = profit + cost` as `profit = total - cost`; both are the same
    relation, chosen by column-name hints (rate/price, profit/net/diff...)."""
    cols = [target, *terms]
    if family == "mul":
        rate = next((c for c in cols if _tokens(c) & _RATE_TOKENS), None)
        if rate is not None:
            num, den = sorted((c for c in cols if c != rate), key=lambda c: (med[c], c), reverse=True)
            if med[rate] <= 1.0:  # a 0-1 rate: numerator is the smaller quantity
                num, den = den, num
            return "ratio", rate, [num, den], f"{_fmt(rate)} = {_fmt(num)} / {_fmt(den)}"
        top = max(cols, key=lambda c: (med[c], c))
        rest = sorted(c for c in cols if c != top)
        return "product", top, rest, f"{_fmt(top)} = " + " * ".join(_fmt(c) for c in rest)
    diff = next((c for c in terms if len(terms) == 2 and _tokens(c) & _DIFF_TOKENS), None)
    if diff is not None:
        other = next(c for c in terms if c != diff)
        return "difference", diff, [target, other], f"{_fmt(diff)} = {_fmt(target)} - {_fmt(other)}"
    kind = "sum" if len(terms) == 2 else "part_of_total"
    return kind, target, terms, f"{_fmt(target)} = " + " + ".join(_fmt(c) for c in terms)


def _cumulative(df: pd.DataFrame, names: list[str], pos: list[int]) -> list[dict[str, Any]]:
    """`total[i] - total[i-1] == step[i]` on the leading rows (row order is
    the ledger order): running totals and lag differences, one identity."""
    cols = np.column_stack([_floats(df.iloc[:_SEQ_ROWS, p]) for p in pos])
    found: list[dict[str, Any]] = []
    for i, target in enumerate(names):
        delta = np.diff(cols[:, i])
        with np.errstate(all="ignore"):
            err = np.abs(delta[:, None] - cols[1:])
        valid = np.isfinite(err)
        scale = float(np.nanmax(np.abs(cols[:, i]))) if np.isfinite(cols[:, i]).any() else 0.0
        for j, step in enumerate(names):
            if j == i or valid[:, j].sum() < MIN_ROWS or scale == 0:
                continue
            e = err[valid[:, j], j]
            d = delta[valid[:, j]]
            if e.max() > _EXACT_ABS * scale or d.min() == d.max():
                continue
            found.append({
                "kind": "cumulative",
                "target": target,
                "terms": [step],
                "expr": f"{_fmt(target)} = previous {_fmt(target)} + {_fmt(step)}",
                "r2": 1.0,
                "max_rel_err": 0.0,
                "exact": True,
                "confidence": "high" if len(e) >= 50 else "medium",
                "n_rows": len(e),
            })
    return found


def find_relations(
    df: pd.DataFrame,
    max_rows_sample: int = 2000,
    exclude: Iterable[str] = (),
    max_relations: int = MAX_RELATIONS,
) -> list[dict[str, Any]]:
    """
    Find columns that are exact or near-exact formulas of other columns.

    Each item: {"kind": product|ratio|sum|difference|part_of_total|cumulative,
    "target", "terms", "expr", "r2", "max_rel_err", "exact", "confidence",
    "n_rows"}. `r2` / `max_rel_err` are measured on the multiplicative or
    additive form the relation was found in, over the full columns (rows where
    every involved column is present). `exact` tolerates float noise and the
    rounding of the target to its own decimals; anything else that clears
    R^2 >= 0.999 and 95% of rows within 5% is "near" (see `max_rel_err`).
    Best relation per target, algebraic equivalents reported once, best first.

    `exclude` names columns that must never take part (identifiers, PII,
    dimensions — the profiler passes its non-measure columns).
    """
    if len(df) < MIN_ROWS or df.shape[1] < 2:
        return []
    n = len(df)
    m = max(MIN_ROWS, max_rows_sample)
    idx = np.arange(n) if n <= m else np.linspace(0, n - 1, m).astype(np.intp)
    names, pos, mat = _candidates(df, idx, {str(c) for c in exclude})
    k = len(names)
    if k < 2:
        return []

    stride = max(1, len(mat) // _TRIPLE_ROWS)
    mat3 = mat[::stride][:_TRIPLE_ROWS]
    with np.errstate(all="ignore"):
        prod2 = mat[:, :, None] * mat[:, None, :]
        sum2 = mat[:, :, None] + mat[:, None, :]
        sum3 = mat3[:, :, None, None] + mat3[:, None, :, None] + mat3[:, None, None, :]
    ar = np.arange(k)
    tri2 = np.triu(np.ones((k, k), dtype=bool), 1)
    tri3 = (ar[:, None, None] < ar[None, :, None]) & (ar[None, :, None] < ar[None, None, :])

    full_cache: dict[int, np.ndarray] = {}

    def full(j: int) -> np.ndarray:
        if j not in full_cache:
            full_cache[j] = _floats(df.iloc[:_FULL_ROWS_CAP, pos[j]])
        return full_cache[j]

    med = {c: float(np.nanmedian(np.abs(mat[:, j]))) for j, c in enumerate(names)}
    found: list[dict[str, Any]] = []
    for i in range(k):
        t = mat[:, i]
        mask2 = tri2.copy()
        mask2[i, :] = mask2[:, i] = False
        tries: list[tuple[float, str, tuple[int, ...]]] = []
        for fam, fit in (("mul", prod2), ("sum", sum2)):
            share = np.where(mask2, _share(t, fit), 0.0)
            for b, c in np.argwhere(share >= _SCREEN_SHARE):
                tries.append((float(share[b, c]), fam, (int(b), int(c))))
        if not any(f == "sum" for _, f, _ in tries):
            mask3 = tri3.copy()
            mask3[i] = False
            mask3[:, i] = False
            mask3[:, :, i] = False
            share3 = np.where(mask3, _share(mat3[:, i], sum3), 0.0)
            for b, c, d in np.argwhere(share3 >= _SCREEN_SHARE):
                tries.append((float(share3[b, c, d]), "sum", (int(b), int(c), int(d))))
            if not any(f == "sum" for _, f, _ in tries) and k >= 4:
                parts = _nnls_parts(mat, i)
                if parts is not None:
                    tries.append((0.0, "sum", parts))
        tries.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, fam, terms in tries[:_PER_TARGET]:
            stats = _confirm(fam, full(i), [full(j) for j in terms])
            if stats is None:
                continue
            kind, target, term_names, expr = _orient(fam, names[i], [names[j] for j in terms], med)
            strong = stats["exact"] or (stats["r2"] >= 0.9995 and stats["max_rel_err"] <= 0.02)
            found.append({
                "kind": kind, "target": target, "terms": term_names, "expr": expr, **stats,
                "confidence": "high" if strong and stats["n_rows"] >= 50 else "medium",
                "_key": (fam, frozenset([names[i], *[names[j] for j in terms]])),
            })
            break  # tries are best-first; one relation per (found) target column
    found += _cumulative(df, names, pos)

    found.sort(key=lambda r: (not r["exact"], len(r["terms"]), -r["r2"], r["target"]))
    result: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    seen_keys: set[Any] = set()
    for rel in found:
        key = rel.pop("_key", None)
        if rel["target"] in seen_targets or (key is not None and key in seen_keys):
            continue
        seen_targets.add(rel["target"])
        if key is not None:
            seen_keys.add(key)
        result.append(rel)
    return result[:max_relations]
