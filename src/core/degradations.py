"""
Observability (item 10) — turns every silent fallback into a visible one.

Each stage upstream (read_any's encoding/delimiter guesses, coerce_types'
repairs, profile_dataframe's sufficiency gate, a failed profiling pass)
already records what it did on its own structured result (ReadReport,
Coercion, DatasetProfile). This module is the single place that flattens
all of those into one human-readable list — "the degradations log" — so
the report and the UI render one thing instead of four separately-shaped
ad-hoc extractions that could drift out of sync with each other.

Pure function of already-computed data — no I/O, no new detection logic.
"""
from __future__ import annotations

from typing import Any


def collect_degradations(
    read_report: dict[str, Any] | None,
    coercions: list[dict[str, Any]] | None,
    profile: dict[str, Any] | None,
    profile_status: str | None,
) -> list[str]:
    """Flatten every recorded fallback/repair/gap into one ordered list."""
    log: list[str] = []

    if isinstance(profile_status, str) and profile_status.startswith("failed"):
        log.append(
            f"Profiling failed ({profile_status[8:].strip()}) — running in degraded mode, "
            "dataset-nature tools (time-series, text, geo...) were unavailable this run."
        )

    if read_report:
        if not read_report.get("encoding_confident", True):
            log.append(f"Encoding was not confidently detected — assumed {read_report.get('encoding')}.")
        if read_report.get("delimiter") and not read_report.get("delimiter_sniffed", False) and read_report.get("format") != "tsv":
            log.append(f"Delimiter assumed as {read_report.get('delimiter')!r} rather than sniffed.")
        for header in read_report.get("duplicate_headers") or []:
            log.append(f"Duplicate column header '{header}' was disambiguated by pandas (e.g. '{header}.1').")
        for note in read_report.get("notes") or []:
            if note not in log:
                log.append(note)

    for c in coercions or []:
        if c.get("rule") == "date_ambiguous":
            # 6.3 — the generic line below reads as a clean repair, but this
            # rule fires precisely when dd/mm vs mm/dd could NOT be
            # determined from the data (no day in the column exceeds 12) —
            # the parser picked one silently. That is the worst failure
            # mode this pipeline can produce (a confidently wrong date axis
            # on every chart), so it gets its own explicit warning instead
            # of blending into the "repaired" phrasing every other rule uses.
            log.append(
                f"Column '{c.get('column')}' — date convention could NOT be determined "
                f"(no day value exceeds 12): {c.get('n_converted')} values parsed using an "
                "assumed day/month order that may be wrong. Verify manually before trusting "
                "any date-based chart or trend for this column."
            )
            continue
        if c.get("rule") == "sentinel":
            log.append(c.get("detail") or f"Column '{c.get('column')}': {c.get('n_failed')} placeholder values treated as missing.")
            continue
        if c.get("is_sentinel_only"):
            failed_ex = c.get("failed_examples")
            examples_str = f" ('{failed_ex[0]}')" if isinstance(failed_ex, (list, tuple)) and failed_ex else ""
            log.append(
                f"Column '{c.get('column')}' repaired from string to {c.get('to_kind')} "
                f"({c.get('rule')} rule): {c.get('n_converted')} converted, {c.get('n_failed')} sentinel/missing values{examples_str} converted to null."
            )
        else:
            log.append(
                f"Column '{c.get('column')}' repaired from string to {c.get('to_kind')} "
                f"({c.get('rule')} rule): {c.get('n_converted')} converted, {c.get('n_failed')} left unparsed."
            )

    if profile:
        if not profile.get("is_sufficient", True):
            log.append(profile.get("sufficiency_reason") or "Insufficient data for a meaningful result.")
        for w in profile.get("warnings") or []:
            if w not in log:
                log.append(w)

    return log
