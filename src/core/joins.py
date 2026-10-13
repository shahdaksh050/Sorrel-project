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

import numpy as np
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

_SYNONYMS = {
    "cust": "customer",
    "prod": "product",
    "acc": "account",
    "usr": "user",
    "org": "organization",
    "emp": "employee",
    "cat": "category",
    "gm": "game",
    "tm": "team",
    "ply": "player",
}


@dataclass
class JoinPlan:
    left_key: str
    right_key: str
    how: str = "left"
    cardinality: str = "many_to_one"   # one_to_one | many_to_one | one_to_many | many_to_many
    coverage: float = 0.0              # share of left rows whose key exists in the right table
    left_name: str = ""
    right_name: str = ""


def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _stem(name: str) -> str:
    n = _norm(Path(name).stem)
    return n[:-3] + "y" if n.endswith("ies") else n.removesuffix("s")


def _canonical_stem(stem: str) -> str:
    return _SYNONYMS.get(stem, stem)


def _is_id_like_col(col_name: str) -> bool:
    norm = _norm(col_name)
    return (
        norm.endswith(("id", "key", "code", "index", "guid", "uuid"))
        or norm in ("id", "key", "code", "index", "guid", "uuid", "pk", "fk")
    )


def _is_integer_valued(s: pd.Series) -> bool:
    """True if series contains integer values (including float64 with NaNs from CSV reading)."""
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        valid = s.dropna()
        if valid.empty:
            return False
        sample = valid if len(valid) <= 1000 else valid.sample(1000, random_state=0)
        arr = sample.to_numpy()
        return bool(np.all(np.floor(arr) == arr))
    return False


def _eligible(s: pd.Series) -> bool:
    """Boolean, datetime, continuous float or near-constant columns are never join keys."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    if pd.api.types.is_float_dtype(s) and not _is_integer_valued(s):
        return False
    s_valid = s.dropna()
    if len(s_valid) < 2:
        return False
    if len(s_valid) > 100_000:
        s_valid = s_valid.sample(100_000, random_state=0)
    if s_valid.nunique() < 2:
        return False
    return float(s_valid.value_counts().iloc[0]) / len(s_valid) <= _NEAR_CONSTANT


def _format_key_series(s: pd.Series) -> pd.Series:
    """Format key series to comparable string values, mapping 101.0 -> '101'."""
    if pd.api.types.is_integer_dtype(s):
        return s.astype("string").str.strip()
    if pd.api.types.is_float_dtype(s) and _is_integer_valued(s):
        valid = s.dropna()
        res = valid.astype(np.int64).astype("string").str.strip()
        out = pd.Series(index=s.index, dtype="string")
        out.loc[res.index] = res
        return out
    return s.astype("string").str.strip()


def _aligned(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Same comparable type on both sides with fallback to case-normalization if helpful."""
    if pd.api.types.is_integer_dtype(a) and pd.api.types.is_integer_dtype(b):
        return a, b
    fa = _format_key_series(a)
    fb = _format_key_series(b)
    va = fa.dropna()
    vb = fb.dropna()
    if not va.empty and not vb.empty:
        sa = va if len(va) <= 1000 else va.sample(1000, random_state=0)
        sb_set = set(vb if len(vb) <= 5000 else vb.sample(5000, random_state=0))
        exact_match = sa.isin(sb_set).mean()
        if exact_match < 0.5:
            sa_low = sa.str.lower()
            sb_low_set = {x.lower() for x in sb_set if isinstance(x, str)}
            if sa_low.isin(sb_low_set).mean() >= 0.6:
                return fa.str.lower(), fb.str.lower()
    return fa, fb


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

    left_unique = (lvals.nunique() / len(lvals)) >= _MIN_RIGHT_UNIQUE
    right_unique = (len(rset) / len(rvals)) >= _MIN_RIGHT_UNIQUE

    if left_unique and right_unique:
        card = "one_to_one"
    elif not left_unique and right_unique:
        card = "many_to_one"
    elif left_unique and not right_unique:
        card = "one_to_many"
    else:
        card = "many_to_many"

    if strict:
        bare = _norm(lc) in _GENERIC_KEYS and _norm(rc) in _GENERIC_KEYS
        if card == "one_to_many":
            ru = pd.Series(rset)
            if len(ru) > _SAMPLE_UNIQUES:
                ru = ru.sample(_SAMPLE_UNIQUES, random_state=0)
            fk_validity = float(ru.isin(pd.unique(lvals)).mean())
            effective_containment = max(containment, fk_validity)
            if effective_containment < (_GENERIC_MIN_CONTAINMENT if bare else _MIN_CONTAINMENT):
                return None
            containment = effective_containment
        else:
            if containment < (_GENERIC_MIN_CONTAINMENT if bare else _MIN_CONTAINMENT):
                return None

    coverage = float(ls.isin(rset).mean())
    return containment, JoinPlan(lc, rc, "left", card, coverage)


def suggest_join(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> JoinPlan | None:
    """Best confident key pair for joining `right` onto `left`, or None."""
    if left.empty or right.empty:
        return None
    stem_r = _stem(right_name)
    canon_r = _canonical_stem(stem_r)
    stem_l = _stem(left_name)
    canon_l = _canonical_stem(stem_l)

    rcols: dict[str, str] = {}
    for c in right.columns:
        rcols.setdefault(_norm(c), c)

    best: tuple[tuple[int, float], JoinPlan] | None = None
    for lc in left.columns:
        nl = _norm(lc)
        if not nl:
            continue
        candidates: list[tuple[str, int]] = []
        if nl in rcols:
            candidates.append((rcols[nl], 3 if nl in _GENERIC_KEYS else 10))
        if nl in (f"{stem_r}id", f"{stem_r}key", f"{stem_r}code", f"{canon_r}id", f"{canon_r}key"):
            for id_name in ("id", "key", "code"):
                if id_name in rcols:
                    candidates.append((rcols[id_name], 9))
        if nl in ("id", "key", "code"):
            for cand_r in (f"{stem_l}id", f"{stem_l}key", f"{canon_l}id", f"{canon_l}key"):
                if cand_r in rcols:
                    candidates.append((rcols[cand_r], 9))
        for nr, rc_name in rcols.items():
            if nr.endswith("id") and nl.endswith(nr) and len(nl) > len(nr):
                candidates.append((rc_name, 7))
            elif nl.endswith("id") and nr.endswith(nl) and len(nr) > len(nl):
                candidates.append((rc_name, 7))

        for rc, priority in candidates:
            hit = _evaluate(left, right, str(lc), str(rc))
            if hit is None:
                continue
            containment, plan = hit
            score = (priority, containment)
            if best is None or score > best[0]:
                best = (score, plan)

    if best is None:
        return None
    return replace(best[1], left_name=left_name, right_name=right_name)


def _aggregate_table_to_key(df: pd.DataFrame, key_col: str, prefix: str) -> pd.DataFrame:
    """Aggregate a child/fact table by key_col so it rolls up to 1 row per key."""
    grouped = df.groupby(key_col, dropna=False)
    count_series = grouped.size()
    count_series.name = f"{prefix}_count"
    res_dfs = [count_series.to_frame()]

    for c in df.columns:
        if c == key_col:
            continue
        col_s = df[c]
        if _is_id_like_col(c):
            agg_df = grouped[c].agg(nunique="nunique").rename(columns={"nunique": f"{prefix}_{c}_nunique"})
            res_dfs.append(agg_df)
        elif pd.api.types.is_bool_dtype(col_s):
            agg_df = grouped[c].agg(share="mean").rename(columns={"share": f"{prefix}_{c}_share"})
            res_dfs.append(agg_df)
        elif pd.api.types.is_numeric_dtype(col_s):
            agg_dict = {f"{prefix}_{c}_mean": "mean"}
            if not any(k in c.lower() for k in ("minute", "hour", "year", "season", "age", "pct", "rate", "ratio", "prob", "rank")):
                agg_dict[f"{prefix}_{c}_sum"] = "sum"
            agg_df = grouped[c].agg(**agg_dict)
            res_dfs.append(agg_df)
        else:
            agg_df = grouped[c].agg(nunique="nunique").rename(columns={"nunique": f"{prefix}_{c}_nunique"})
            res_dfs.append(agg_df)

    out = pd.concat(res_dfs, axis=1).reset_index()
    return out


def join_frames(left: pd.DataFrame, right: pd.DataFrame, plan: JoinPlan, max_rows: int) -> tuple[pd.DataFrame, list[str]]:
    """Merge `right` onto `left` per `plan`; returns (frame, notes)."""
    name = Path(plan.right_name).stem or "related"
    card_label = plan.cardinality.replace("_", "-")
    was_aggregated = False
    r_to_merge = right

    should_aggregate = (
        plan.cardinality == "many_to_many"
        or (
            plan.cardinality == "one_to_many"
            and (
                len(right) > 20_000
                or (len(left) > 0 and len(right) / len(left) > 5.0 and len(right) > 5_000)
            )
        )
    )
    if should_aggregate:
        r_to_merge = _aggregate_table_to_key(right, plan.right_key, name)
        was_aggregated = True

    lkey, rkey = _aligned(left[plan.left_key], r_to_merge[plan.right_key])
    r = r_to_merge.drop(columns=[plan.right_key]).assign(**{_TMP_KEY: rkey.to_numpy()})
    r = r[rkey.notna().to_numpy()]
    if plan.cardinality in ("many_to_one", "one_to_one") or was_aggregated:
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
    left_stem_name = Path(plan.left_name).stem or "the main table"
    if was_aggregated:
        notes = [
            f"Joined {name} to {left_stem_name} on {on} "
            f"(aggregated child table to avoid Cartesian fan-out, {len(right):,} records rolled up to {len(r_to_merge):,} entities, {plan.coverage:.0%} of keys matched)."
        ]
    else:
        notes = [
            f"Joined {name} to {left_stem_name} on {on} "
            f"({card_label}, {plan.coverage:.0%} of rows matched)."
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
            prev_cols = set(map(str, joined.columns))
            joined, jn = join_frames(joined, df, plan, max_rows)
            notes += jn
            new_cols = set(map(str, joined.columns)) - prev_cols
            owners[name] = new_cols | (set(map(str, df.columns)) - {plan.right_key})
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
