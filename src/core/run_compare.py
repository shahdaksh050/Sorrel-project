"""
Compare two run summaries (reports/summary.json) — pure functions, no I/O
beyond `list_runs`. Findings match on finding_id, falling back to the headline
with digits stripped (so "Sales up 12%" and "Sales up 15%" are the same claim).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

IMPORTANCE_DELTA = 0.1


def _headline_key(headline: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+(?:[.,]\d+)*", "", str(headline))).strip().lower()


def _match(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(matched pairs, removed from old, added in new)."""
    by_id = {f["finding_id"]: f for f in old if f.get("finding_id")}
    by_head: dict[str, dict[str, Any]] = {}
    for f in old:
        by_head.setdefault(_headline_key(f.get("headline", "")), f)
    used: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    added: list[dict[str, Any]] = []
    for f in new:
        o = by_id.get(f.get("finding_id")) if f.get("finding_id") else None
        if o is None or id(o) in used:
            o = by_head.get(_headline_key(f.get("headline", "")))
        if o is None or id(o) in used:
            added.append(f)
            continue
        used.add(id(o))
        pairs.append((o, f))
    return pairs, [f for f in old if id(f) not in used], added


def compare_runs(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Diff `new` against `old`: findings added/removed/changed, charts
    added/removed, and per-tool time deltas (seconds, new - old)."""
    pairs, removed, added = _match(old.get("findings") or [], new.get("findings") or [])
    changed = []
    for o, n in pairs:
        delta = float(n.get("importance") or 0.0) - float(o.get("importance") or 0.0)
        if abs(delta) > IMPORTANCE_DELTA:
            changed.append({
                "headline": n.get("headline", ""), "old": float(o.get("importance") or 0.0),
                "new": float(n.get("importance") or 0.0), "delta": delta,
            })
    changed.sort(key=lambda c: -abs(c["delta"]))

    def chart_map(run: dict[str, Any]) -> dict[str, str]:
        return {str(c.get("chart_id")): str(c.get("title", "")) for c in run.get("charts") or []}

    oc, nc = chart_map(old), chart_map(new)
    ot, nt = old.get("tool_seconds") or {}, new.get("tool_seconds") or {}
    times = [
        {"tool": t, "old": float(ot.get(t, 0.0)), "new": float(nt.get(t, 0.0)),
         "delta": float(nt.get(t, 0.0)) - float(ot.get(t, 0.0))}
        for t in sorted(set(ot) | set(nt))
    ]
    times.sort(key=lambda r: -abs(r["delta"]))
    return {
        "findings_added": [f.get("headline", "") for f in added],
        "findings_removed": [f.get("headline", "") for f in removed],
        "findings_changed": changed,
        "charts_added": [t for k, t in nc.items() if k not in oc],
        "charts_removed": [t for k, t in oc.items() if k not in nc],
        "tool_time_deltas": times,
    }


def list_runs(root: str | Path = "output/runs") -> list[dict[str, Any]]:
    """Summaries under `root/*/reports/summary.json`, newest first, each with
    `path` and a human `label` (dataset + time)."""
    runs: list[dict[str, Any]] = []
    for p in Path(root).glob("*/reports/summary.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        created = str(data.get("created") or "")
        name = (data.get("dataset") or {}).get("name", p.parents[1].name)
        data["path"] = str(p)
        data["label"] = f"{name} · {created.replace('T', ' ')[:16] or p.parents[1].name}"
        data["_mtime"] = p.stat().st_mtime
        runs.append(data)
    runs.sort(key=lambda r: r["_mtime"], reverse=True)
    return runs
