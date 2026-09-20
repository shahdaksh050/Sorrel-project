"""
Join related tables (orders + customers + products) into one analysis frame.

Keys are inferred, never guessed blindly: a candidate needs a matching
(normalised) name, the left key's values must mostly exist in the right key, and
the right key must be ~unique so the join cannot multiply rows. Anything less
confident is skipped and named in a note. Pandas-vectorised; a run with no
related tables never reaches this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

_SAMPLE_UNIQUES = 5_000
_MIN_CONTAINMENT = 0.6
_MIN_RIGHT_UNIQUE = 0.99
_MAX_MANY_TO_MANY_GROWTH = 1.05
_NEAR_CONSTANT = 0.95
_TMP_KEY = "__dsa_join_key__"
_IDLIKE = ("id", "key", "code")
_GENERIC_KEYS = frozenset({"id", "index", "code", "key"})   # bare names any table may have
_GENERIC_MIN_CONTAINMENT = 0.9


@dataclass
class JoinPlan:
    left_key: str
    right_key: str
    how: str = "left"
    cardinality: str = "many_to_one"   # one_to_one | many_to_one | many_to_many
    coverage: float = 0.0              # share of left rows whose key exists in the right table
    left_name: str = ""
    right_name: str = ""


def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _stem(name: str) -> str:
    n = _norm(Path(name).stem)
    return n[:-3] + "y" if n.endswith("ies") else n.removesuffix("s")


def _eligible(s: pd.Series) -> bool:
    """Float, boolean, datetime and near-constant columns are never join keys."""
    if (pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_datetime64_any_dtype(s)):
        return False
    s = s.dropna()
    if len(s) > 100_000:
        s = s.sample(100_000, random_state=0)
    if s.nunique() < 2:
        return False
    return float(s.value_counts().iloc[0]) / len(s) <= _NEAR_CONSTANT


def _aligned(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Same comparable type on both sides (an int id vs a text id would not merge)."""
    if pd.api.types.is_integer_dtype(a) and pd.api.types.is_integer_dtype(b):
        return a, b
    return a.astype("string").str.strip(), b.astype("string").str.strip()


def _evaluate(
    left: pd.DataFrame, right: pd.DataFrame, lc: str, rc: str, strict: bool = True
) -> tuple[float, JoinPlan] | None:
    """Score a key pair. `strict=False` (user-chosen keys) skips the confidence gates."""
    if strict and not (_eligible(left[lc]) and _eligible(right[rc])):
        return None
    ls, rs = _aligned(left[lc], right[rc])
    rvals = rs.dropna()
    lvals = ls.dropna()
    if rvals.empty or lvals.empty:
        return None
    rset = pd.unique(rvals)
    lu = pd.Series(pd.unique(lvals))
    if len(lu) > _SAMPLE_UNIQUES:
        lu = lu.sample(_SAMPLE_UNIQUES, random_state=0)
    containment = float(lu.isin(rset).mean())
    if strict:
        bare = _norm(lc) in _GENERIC_KEYS and _norm(rc) in _GENERIC_KEYS
        if containment < (_GENERIC_MIN_CONTAINMENT if bare else _MIN_CONTAINMENT):
            return None
    if len(rset) / len(rvals) >= _MIN_RIGHT_UNIQUE:
        one = lvals.nunique() / len(lvals) >= _MIN_RIGHT_UNIQUE
        card = "one_to_one" if one else "many_to_one"
    else:
        grown = float(ls.map(rvals.value_counts()).fillna(1).clip(lower=1).sum())
        if strict and grown > _MAX_MANY_TO_MANY_GROWTH * len(left):
            return None
        card = "many_to_many"
    return containment, JoinPlan(lc, rc, "left", card, float(ls.isin(rset).mean()))


def suggest_join(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> JoinPlan | None:
    """Best confident key pair for joining `right` onto `left`, or None."""
    if left.empty or right.empty:
        return None
    stem = _stem(right_name)
    rcols: dict[str, str] = {}
    for c in right.columns:
        rcols.setdefault(_norm(c), c)
    best: tuple[tuple[bool, float], JoinPlan] | None = None
    for lc in left.columns:
        n = _norm(lc)
        if not n:
            continue
        rc = rcols.get(n) if n != "id" else None
        if rc is None and "id" in rcols and n in (stem + "id", stem + "key"):
            rc = rcols["id"]   # orders.customer_id <-> customers.id
        if rc is None:
            continue
        hit = _evaluate(left, right, str(lc), str(rc))
        if hit is None:
            continue
        rank = (n.endswith(_IDLIKE), hit[0])
        if best is None or rank > best[0]:
            best = (rank, hit[1])
    if best is None:
        return None
    return replace(best[1], left_name=left_name, right_name=right_name)


def join_frames(left: pd.DataFrame, right: pd.DataFrame, plan: JoinPlan, max_rows: int) -> tuple[pd.DataFrame, list[str]]:
    """Merge `right` onto `left` per `plan`; returns (frame, notes)."""
    name = Path(plan.right_name).stem or "related"
    lkey, rkey = _aligned(left[plan.left_key], right[plan.right_key])
    r = right.drop(columns=[plan.right_key]).assign(**{_TMP_KEY: rkey.to_numpy()})
    r = r[rkey.notna().to_numpy()]
    if plan.cardinality != "many_to_many":
        r = r.drop_duplicates(_TMP_KEY)
    taken = set(left.columns)
    rename: dict[str, str] = {}
    for c in r.columns:
        if c != _TMP_KEY and c in taken:
            new = f"{name}_{c}"
            while new in taken or new in rename.values():
                new += "_"
            rename[c] = new
    out = left.assign(**{_TMP_KEY: lkey.to_numpy()}).merge(
        r.rename(columns=rename), on=_TMP_KEY, how=plan.how
    ).drop(columns=[_TMP_KEY])
    on = plan.left_key if plan.left_key == plan.right_key else f"{plan.left_key} = {plan.right_key}"
    notes = [
        f"Joined {name} to {Path(plan.left_name).stem or 'the main table'} on {on} "
        f"({plan.cardinality.replace('_', '-')}, {plan.coverage:.0%} of rows matched)."
    ]
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=0).sort_index()
        notes.append(f"Joined table exceeded {max_rows:,} rows; a random sample was kept.")
    return out.reset_index(drop=True), notes


def _override_plan(
    joined: pd.DataFrame, df: pd.DataFrame, ov: dict[str, Any], base_name: str, name: str
) -> JoinPlan | None:
    lk, rk = ov.get("left_key"), ov.get("right_key")
    if lk not in joined.columns or rk not in df.columns:
        return None
    hit = _evaluate(joined, df, str(lk), str(rk), strict=False)
    return None if hit is None else replace(hit[1], left_name=base_name, right_name=name)


def join_related(
    base: pd.DataFrame,
    related: list[tuple[str, pd.DataFrame]],
    max_rows: int,
    base_name: str = "main table",
    overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Greedily attach each related table to the base or an already-joined table.
    Returns `base` itself (same object) when nothing could be joined.

    `overrides[name]` may be {"skip": True} or {"left_key", "right_key"} (the
    user's review choice, honoured without the confidence gates).
    """
    overrides = overrides or {}
    joined = base
    notes: list[str] = []
    owners = {base_name: set(map(str, base.columns))}
    pending = []
    for item in related:
        if overrides.get(item[0], {}).get("skip"):
            notes.append(f"Skipped {item[0]}: you chose not to join it.")
        else:
            pending.append(item)
    progress = True
    while pending and progress:
        progress = False
        for item in list(pending):
            name, df = item
            ov = overrides.get(name) or {}
            if ov.get("left_key") and ov.get("right_key"):
                plan = _override_plan(joined, df, ov, base_name, name)
            else:
                plan = suggest_join(joined, df, base_name, name)
            if plan is None:
                continue
            plan.left_name = next((n for n, cols in owners.items() if plan.left_key in cols), base_name)
            joined, jn = join_frames(joined, df, plan, max_rows)
            notes += jn
            owners[name] = set(map(str, df.columns)) - {plan.right_key}
            pending.remove(item)
            progress = True
    notes += [f"Skipped {n}: no confident join key found." for n, _ in pending]
    return joined, notes


def preview_joins(
    base: pd.DataFrame | str | Path,
    related: list[tuple[str, pd.DataFrame | str | Path]],
    base_name: str = "main table",
) -> list[dict[str, Any]]:
    """
    What `join_related` would propose, without merging. Per related table:
    {name, plan (left_key, right_key, cardinality, coverage, left_name) or None,
    reason_if_none, columns (its own), left_columns (candidate left keys)}.
    A table is tried against the base first, then against the other related tables.
    """
    def load(x: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x
        from src.core.io import read_any
        return read_any(str(x))[0]

    base_df = load(base)
    frames = [(n, load(d)) for n, d in related]
    out: list[dict[str, Any]] = []
    for name, df in frames:
        lefts = [(base_name, base_df)] + [(n, d) for n, d in frames if n != name]
        plan = None
        for lname, ldf in lefts:
            plan = suggest_join(ldf, df, lname, name)
            if plan:
                break
        out.append({
            "name": name,
            "plan": None if plan is None else {
                "left_key": plan.left_key, "right_key": plan.right_key,
                "cardinality": plan.cardinality, "coverage": plan.coverage,
                "left_name": plan.left_name,
            },
            "reason_if_none": "" if plan else (
                "The table is empty." if df.empty or base_df.empty
                else "No shared ID column with enough matching values."),
            "columns": [str(c) for c in df.columns],
            "left_columns": [str(c) for c in base_df.columns]
            + [str(c) for n, d in frames if n != name for c in d.columns],
        })
    return out
