"""
Verbatim-claim verification — flags numbers in the LLM's final synthesis
that cannot be traced back to an actual tool result.

Split out of ``src/core/controller.py``; ``AgentController`` keeps thin
``_verified_number_pool(s)`` / ``_flag_unverified_claims`` methods that
delegate here.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.memory import MemorySystem

# ---------------------------------------------------------------------------
# Verbatim-metric validation (P0.7) — SYSTEM_PROMPT_CORE tells the LLM to
# cite only numbers that appear in tool results, but nothing checked that
# rule. These turn it into a mechanism: any numeric literal the LLM's
# synthesis states that cannot be traced back to an actual tool result is
# flagged, not trusted silently.
# ---------------------------------------------------------------------------

#: Matches numeric literals (integers, decimals, negatives, comma-formatted) in free text.
_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

#: Single-digit integers are almost always counts ("3 models", "top 5
#: features") rather than cited metrics, and are cheap to satisfy by
#: coincidence — excluding them keeps the flag meaningful.
_UNVERIFIABLE_SKIP_ABS_INT = 9

# ---------------------------------------------------------------------------
# Attribution-aware verification — a number existing ANYWHERE in the
# pooled tool output isn't enough: "cleaning handled 503 missing values"
# verified as long as 503 appeared in ANY tool's output, even when the real
# clean_data result said "cleaned 0 missing values" and 503 was actually
# ingest_dataset's row count. When an insight/recommendation sentence names
# a specific kind of analysis, its numbers must come from the tool(s) that
# analysis maps to, not merely from the run somewhere.
# ---------------------------------------------------------------------------

#: Keyword -> the tool name(s) whose own output pool a sentence containing
#: that keyword must be checked against. Each key is a regex matched with a
#: leading word boundary against the lower-cased claim text (so "chi-square"
#: matches but "which" doesn't, and a prefix like "correlat" still covers
#: "correlation"/"correlated"); a sentence can match several keywords/tools
#: at once, in which case the union of their pools is used. Deliberately
#: specific — a generic word ("test", "segment" alone) would pull ordinary
#: prose into the stricter per-tool check and flag correct numbers.
_KEYWORD_TOOL_MAP: dict[str, tuple[str, ...]] = {
    r"clean": ("clean_data",),
    r"imput": ("clean_data",),
    r"outlier": ("detect_outliers",),
    r"correlat": ("correlation_analysis",),
    r"pca\b": ("dimensionality_analysis",),
    r"principal component": ("dimensionality_analysis",),
    r"dimensionality": ("dimensionality_analysis",),
    r"cluster": ("cluster_data",),
    r"silhouette": ("cluster_data",),
    r"concentrat": ("concentration_analysis",),
    r"gini": ("concentration_analysis",),
    r"segments?\b": ("segment_comparison", "cluster_data"),
    r"trend": ("time_series_analysis", "change_analysis"),
    r"seasonal": ("time_series_analysis", "change_analysis"),
    r"accuracy": ("train_model", "evaluate_model"),
    r"f1\b": ("train_model", "evaluate_model"),
    r"auc\b": ("train_model", "evaluate_model"),
    r"r2\b": ("train_model", "evaluate_model"),
    r"statistical test": ("select_statistical_test", "segment_comparison"),
    r"p-?value": ("select_statistical_test", "segment_comparison"),
    r"anova": ("select_statistical_test",),
    r"chi-?squared?\b": ("select_statistical_test",),
    r"mann-whitney": ("select_statistical_test",),
    r"kruskal": ("select_statistical_test",),
}
_KEYWORD_TOOL_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(r"\b" + kw), tools) for kw, tools in _KEYWORD_TOOL_MAP.items()
)


def _canon_number(value: Any, precision: int = 4) -> str:
    """Normalise a number to a fixed-precision canonical string for
    set-membership comparison, so '0.8', '0.80' and 0.7999999999999999
    (float round-trip noise) all match."""
    try:
        if isinstance(value, str):
            value = value.replace(",", "")
        f = float(value)
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        return f"{round(f, precision):g}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


#: Finding headlines (7.1) round for readability (e.g. "r=0.81") while the
#: tool output they're traced back to often carries more decimals
#: ("correlation=0.8109") — the verified pool indexes several roundings of
#: each source number so a claim's own (looser) precision still matches
#: without weakening the check itself (a genuinely wrong number still fails
#: at every precision).
_CANON_PRECISIONS = (4, 3, 2, 1, 0)


def _collect_numbers(obj: Any, into: set[str]) -> None:
    """Recursively flatten every numeric leaf/substring in a JSON-like
    structure into canonical form, at several roundings (see
    `_CANON_PRECISIONS`)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        for p in _CANON_PRECISIONS:
            into.add(_canon_number(obj, p))
        # Finding headlines and narrative text display rates/fractions as
        # percentages (e.g. evidence["level_value"]=0.08596 renders as
        # "8.6%"), while the raw tool output stores the fraction — a unit
        # mismatch, not a rounding one, since _CANON_PRECISIONS already
        # covers rounding (that's why "r=0.81" verifies against a stored
        # 0.8109 but "8.6%" didn't verify against a stored 0.08596). This
        # function sees only values, not keys, so it can't check the
        # evidence dict's own `is_rate` flag — a numeric range guard is the
        # generic equivalent: only fraction-range values get a *100 form,
        # so this can't turn an unrelated large number into a false
        # verification.
        if -1.0 <= obj <= 1.0:
            for p in _CANON_PRECISIONS:
                into.add(_canon_number(obj * 100, p))
    elif isinstance(obj, str):
        for match in _NUMBER_RE.finditer(obj):
            val = match.group().replace(",", "")
            for p in _CANON_PRECISIONS:
                into.add(_canon_number(val, p))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, into)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, into)


def verified_number_pools(memory: MemorySystem) -> tuple[set[str], dict[str, set[str]]]:
    """Global pool (unchanged — every numeric literal across all tool
    results) plus a pool per tool_name, so a claim keyword-attributed to
    a specific kind of analysis (`_KEYWORD_TOOL_MAP`) can be checked
    against only that tool's OWN output rather than the whole run's
    pooled numbers — a row count from ingest_dataset's summary must not
    verify a claim about what clean_data did."""
    global_pool: set[str] = set()
    per_tool: dict[str, set[str]] = {}
    for r in memory.tool_results:
        tool_pool = per_tool.setdefault(r.tool_name, set())
        _collect_numbers(r.to_dict(), tool_pool)
        global_pool |= tool_pool
    if memory.dataset_metadata:
        meta_pool: set[str] = set()
        _collect_numbers(memory.dataset_metadata.__dict__, meta_pool)
        global_pool |= meta_pool
    # Findings are computed by tool code from tool output (ratios, shares,
    # differences), and the synthesis prompt tells the LLM to copy their
    # numbers — a derived headline number is verified, not invented.
    # Scope of the guarantee: `custom_analysis` findings come from
    # LLM-authored sandbox code, so their numbers are verified as "computed
    # by code that actually ran on the data", not as "produced by a
    # built-in tool". This check catches numbers invented at synthesis
    # time; the audit log (governance.py) is the record of what that code was.
    for f in memory.findings:
        finding_pool: set[str] = set()
        _collect_numbers([f.headline, f.detail, f.evidence, f.effect], finding_pool)
        per_tool.setdefault(f.source_tool, set()).update(finding_pool)
        global_pool |= finding_pool
    return global_pool, per_tool


def flag_unverified_claims(memory: MemorySystem, final_result: dict[str, Any]) -> list[str]:
    """
    Enforce SYSTEM_PROMPT_CORE's "cite only verbatim metrics" rule.

    Any numeric literal in `insights`/`recommendations`/`key_metrics`
    that doesn't trace back to a real tool result is annotated
    in-place with `[unverified: ...]` (never silently trusted) and
    returned so the caller can log/report the hallucination rate.

    Round 8 hardening — a number existing ANYWHERE in the pooled tool
    output used to be enough to "verify" it, which passed claims that
    cited the right number from the WRONG tool (e.g. citing
    ingest_dataset's row count as clean_data's missing-value count,
    since both numbers were in the pool). When a sentence names a
    specific kind of analysis (`_KEYWORD_TOOL_MAP`) and that analysis's
    tool(s) actually ran, its numbers are checked against only that
    tool's own output; a sentence with no such keyword keeps the
    original global-pool check.
    """
    verified, per_tool_pools = verified_number_pools(memory)
    ran_tools = {r.tool_name for r in memory.tool_results}
    flagged: list[str] = []

    meta = memory.dataset_metadata
    meta_pool: set[str] = set()
    if meta:
        for p in _CANON_PRECISIONS:
            meta_pool.add(_canon_number(meta.row_count, p))
            meta_pool.add(_canon_number(meta.column_count, p))

    for field in ("insights", "recommendations"):
        items = final_result.get(field)
        if not isinstance(items, list):
            continue
        for i, item in enumerate(items):
            if not isinstance(item, str):
                continue
            claimed = [
                m.group() for m in _NUMBER_RE.finditer(item)
                if not (
                    "." not in m.group()
                    and abs(int(m.group().replace(",", ""))) <= _UNVERIFIABLE_SKIP_ABS_INT
                )
            ]
            if not claimed:
                continue

            item_l = item.lower()
            attributed_tools = sorted({
                tool
                for pattern, tools in _KEYWORD_TOOL_PATTERNS
                if pattern.search(item_l)
                for tool in tools
                if tool in ran_tools
            })
            if attributed_tools:
                pool = set().union(*(per_tool_pools.get(t, set()) for t in attributed_tools))
                valid_pool = pool | meta_pool
                bad = sorted({n for n in claimed if _canon_number(n) not in valid_pool})
                if bad:
                    attribution = "/".join(attributed_tools)
                    items[i] = (
                        f"{item} [unverified: {', '.join(bad)} "
                        f"(not in {attribution} output)]"
                    )
                    flagged.append(
                        f"{field}[{i}]: {', '.join(bad)} (attributed to {attribution})"
                    )
                continue

            bad = sorted({n for n in claimed if _canon_number(n) not in verified})
            if bad:
                items[i] = f"{item} [unverified: {', '.join(bad)}]"
                flagged.append(f"{field}[{i}]: {', '.join(bad)}")

    key_metrics = final_result.get("key_metrics")
    if isinstance(key_metrics, dict):
        for k, v in key_metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nums = [str(v)]
            elif isinstance(v, str):
                nums = [m.group() for m in _NUMBER_RE.finditer(v)]
            else:
                continue
            bad = sorted({n for n in nums if _canon_number(n) not in verified})
            if bad:
                flagged.append(f"key_metrics.{k}={v!r} [unverified: {', '.join(bad)}]")

    return flagged
