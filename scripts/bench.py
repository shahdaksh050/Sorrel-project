"""
Performance budget for the deterministic (--no-llm) pipeline.

    python scripts/bench.py                    # run + print table
    python scripts/bench.py --check            # exit 1 on regression vs bench_baseline.json
    python scripts/bench.py --update-baseline  # rewrite bench_baseline.json
    python scripts/bench.py --only wide300 --json

A metric regresses when it exceeds baseline by > 30% AND > 0.5 s absolute.
Baselines are machine-specific (see the "machine" note in the JSON).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASELINE = Path(__file__).with_name("bench_baseline.json")
REL_TOL, ABS_TOL = 0.30, 0.5


# ---------------------------------------------------------------- datasets
def _air_quality(_: Path) -> Path | None:
    p = ROOT / "data" / "AirQualityUCI.csv"
    return p if p.exists() else None


def _big500k(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 500_000
    df = pd.DataFrame({
        "order_id": np.arange(n),
        "ts": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365 * 24 * 60, n), unit="m"),
        "amount": rng.lognormal(3, 1, n).round(2),
        "quantity": rng.integers(1, 10, n),
        "price": rng.normal(50, 12, n).round(2),
        "discount": rng.uniform(0, 0.3, n).round(3),
        "score": rng.normal(0, 1, n),
        "customer_id": rng.integers(0, 50_000, n).astype(str),   # high cardinality
        "region": rng.choice(list("NSEW"), n),
        "channel": rng.choice(["web", "store", "app"], n),
        "segment": rng.choice(["A", "B", "C", "D", "E"], n),
        "category": rng.choice([f"cat{i}" for i in range(12)], n),
        "status": rng.choice(["ok", "returned", "cancelled"], n, p=[0.9, 0.06, 0.04]),
    })
    df["revenue"] = (df["quantity"] * df["price"]).round(2)
    p = tmp / "big500k.csv"
    df.to_csv(p, index=False)
    return p


def _wide300(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    n, k = 20_000, 300
    x = rng.normal(size=(n, k))
    for j in range(1, 6):                       # a few correlated columns
        x[:, j] = x[:, 0] * 0.9 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame(x, columns=[f"f{i:03d}" for i in range(k)]).round(4)
    for c in ("f100", "f150", "f200"):          # sentinel planted in 3 columns
        df.loc[rng.choice(n, 400, replace=False), c] = -999
    p = tmp / "wide300.csv"
    df.to_csv(p, index=False)
    return p


def _mixed60(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 20_000
    cols: dict[str, Any] = {}
    for i in range(25):
        cols[f"num{i}"] = rng.normal(100 * i, 10 + i, n).round(3)
    for i in range(8):
        cols[f"cnt{i}"] = rng.poisson(5, n)
    for i in range(20):
        cols[f"cat{i}"] = rng.choice([f"v{j}" for j in range(3 + i)], n)
    for i in range(4):
        cols[f"flag{i}"] = rng.integers(0, 2, n).astype(bool)
    for i in range(2):
        cols[f"date{i}"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    words = np.array("late great broken slow fast love hate refund support delivery quality price".split())
    cols["review"] = [" ".join(w) for w in words[rng.integers(0, len(words), (n, 8))]]
    p = tmp / "mixed60.csv"
    pd.DataFrame(cols).to_csv(p, index=False)
    return p


DATASETS: dict[str, tuple[Callable[[Path], Path | None], int]] = {
    "airquality": (_air_quality, 2),
    "big500k": (_big500k, 1),
    "wide300": (_wide300, 2),
    "mixed60": (_mixed60, 2),
}


# ---------------------------------------------------------------- measuring
def _t(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def _run_once(path: Path) -> dict[str, float]:
    from src.core.coercion import coerce_types
    from src.core.controller import AgentController
    from src.core.dashboard import build_dashboard
    from src.core.io import read_any
    from src.core.profiler import clear_profile_cache, profile_dataframe
    from src.core.relations import find_relations
    from src.core.sentinels import null_sentinels
    from src.core.tool_registry import ToolRegistry

    m: dict[str, float] = {}
    clear_profile_cache()   # profile_dataframe memoises; measure cold
    with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
        os.environ["OUTPUT_DIR"] = out
        agent = AgentController(use_llm=False, enable_rlm=False)
        t0 = time.perf_counter()
        agent.load_dataset(str(path), interactive=False)
        agent.analyze()
        m["total"] = time.perf_counter() - t0
        for r in agent.memory.tool_results:
            k = f"tool.{r.tool_name}"
            m[k] = m.get(k, 0.0) + r.execution_time_ms / 1000.0

        # isolated hot-spot timings on the loaded frame
        df, rep = read_any(str(path))
        df, _ = coerce_types(df, delimiter=rep.delimiter)
        clear_profile_cache()
        m["stage.profile_dataframe"], profile = _t(lambda: profile_dataframe(df))
        m["stage.find_relations"], _ = _t(lambda: find_relations(df))
        m["stage.null_sentinels"], _ = _t(lambda: null_sentinels(df))

        def cands() -> list[Any]:
            reg = ToolRegistry()
            return reg.candidate_tools(profile, None)

        m["stage.candidate_tools"], _ = _t(cands)
        m["stage.build_dashboard"], _ = _t(lambda: build_dashboard(df, profile))
    return m


def bench_dataset(name: str, tmp: Path) -> dict[str, float] | None:
    make, reps = DATASETS[name]
    path = make(tmp)
    if path is None:
        print(f"[skip] {name}: dataset file missing", file=sys.stderr)
        return None
    best: dict[str, float] = {}
    for i in range(reps):
        print(f"[run] {name} {i + 1}/{reps}", file=sys.stderr, flush=True)
        for k, v in _run_once(path).items():
            best[k] = min(best.get(k, v), v)
    return {f"{name}.{k}": round(v, 3) for k, v in best.items()}


def machine_note() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail on regression vs baseline")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--only", choices=list(DATASETS), help="run a single dataset")
    ap.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = ap.parse_args()

    names = [args.only] if args.only else list(DATASETS)
    metrics: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n in names:
            metrics.update(bench_dataset(n, Path(tmp)) or {})

    if args.update_baseline:
        old: dict[str, Any] = {}
        if BASELINE.exists():
            old = json.loads(BASELINE.read_text(encoding="utf-8")).get("metrics", {})
        old.update(metrics)   # --only refreshes just its own keys
        BASELINE.write_text(json.dumps({"machine": machine_note(), "metrics": dict(sorted(old.items()))}, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE}", file=sys.stderr)

    base: dict[str, float] = {}
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text(encoding="utf-8")).get("metrics", {})

    failed: list[str] = []
    rows = []
    for k, v in metrics.items():
        b = base.get(k)
        bad = b is not None and v > b * (1 + REL_TOL) and v - b > ABS_TOL
        if bad:
            failed.append(k)
        rows.append((k, v, b, bad))

    if args.json:
        print(json.dumps({"machine": machine_note(), "metrics": metrics, "regressions": failed}, indent=2))
    else:
        w = max((len(r[0]) for r in rows), default=10)
        print(f"{'metric':<{w}}  {'now(s)':>8}  {'base(s)':>8}  {'delta':>7}")
        for k, v, b, bad in rows:
            if b is None:
                print(f"{k:<{w}}  {v:>8.3f}  {'-':>8}")
            else:
                d = (v / b - 1) * 100 if b else 0.0
                print(f"{k:<{w}}  {v:>8.3f}  {b:>8.3f}  {d:>+6.0f}%{'  REGRESSION' if bad else ''}")

    if args.check and failed:
        print(f"\nFAIL: {len(failed)} metric(s) over budget (>{REL_TOL:.0%} and >{ABS_TOL}s): {', '.join(failed)}", file=sys.stderr)
        return 1
    if args.check:
        print("\nOK: within budget", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
