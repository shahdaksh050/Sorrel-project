"""
Audited-entry check marks — FrontendPlan.md section 5.

The plain-language rendering of `Finding.evidence["checks"]`, shared by every
surface that shows a finding: `ui/components/cards.py`'s Streamlit cards,
`src/core/html_report.py`'s static report, and the Charts tab (DB11). Only
this module decides which checks appear, in what order, with what label and
state; each surface still owns its own markup and CSS — a Streamlit-injected
`<div class="...">` and a standalone report document are different enough
contexts that sharing raw HTML between them would cost more than it saves.
What must not fork three ways is the *logic*: which finding gets a "Not
luck" tick, and what "42 records" reads as when it fails.

`src.core.findings.attach_finding_checks` is the only writer of
`evidence["checks"]`, from audits that already ran in the pipeline. This
module only reads that dict — it never re-derives a verdict from raw numbers,
and a check key absent from `checks` produces no `CheckMark` at all, never a
fake tick (FrontendPlan.md risk #2: "an unrun check must never render
nothing... a tick for a test that was not done would betray the product's
promise" — the risk doc's own wording is inverted, but the rule is the one
implemented here: absent means nothing renders).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CheckState = Literal["pass", "fail", "neutral"]

#: Rendering glyph per state, per the FrontendPlan.md section 5 mockup
#: ("✓ Not luck ... ! Only 42 records ... ○ Pattern, not proof of cause").
#: Plain Unicode marks, not emoji — every mark also carries a text label, so
#: colour/glyph alone never carries the meaning (section 5, "Design rules").
GLYPH: dict[CheckState, str] = {"pass": "✓", "fail": "!", "neutral": "○"}

#: Rendering order for a finding's check row.
CHECK_ORDER: tuple[str, ...] = ("not_luck", "records", "robust", "leakage", "cause", "model_gap")

_SAMPLE_KEYS = ("n", "n_level", "sample_size", "n_segment", "n_obs")

#: key -> (pass label, fail label). "leakage" and "cause" are handled
#: specially below (leakage is fail-only; cause is neutral-only).
_LABELS: dict[str, tuple[str, str]] = {
    "not_luck": ("Not luck", "Could be chance"),
    "robust": ("Holds without extreme rows", "Depends on a few rows"),
    "model_gap": ("Holds on new data", "May not hold on new data"),
}


@dataclass(frozen=True, slots=True)
class CheckMark:
    key: str
    label: str
    state: CheckState


def _records_label(evidence: dict[str, Any], passed: bool) -> str:
    sizes = [
        v for k in _SAMPLE_KEYS
        if isinstance(v := evidence.get(k), (int, float)) and not isinstance(v, bool)
    ]
    if not sizes:
        return "Enough records" if passed else "Few records"
    n = int(min(sizes))
    return f"{n:,} records" if passed else f"Only {n:,} records"


def audited_checks(evidence: dict[str, Any] | None) -> list[CheckMark]:
    """Ordered `CheckMark` list for one finding's `evidence["checks"]`.

    `evidence` is a finding's raw evidence dict (e.g. `finding_dict["evidence"]`
    from `Finding.to_dict()`) — a dict with no `"checks"` key, or that isn't a
    dict at all, yields an empty list, not an error: a finding this ran
    against but got no data for shows no check row, same as one the audits
    never touched.
    """
    if not isinstance(evidence, dict):
        return []
    checks = evidence.get("checks")
    if not isinstance(checks, dict):
        return []

    marks: list[CheckMark] = []
    for key in CHECK_ORDER:
        if key not in checks:
            continue
        value = checks[key]

        if key == "records":
            passed = bool(value)
            marks.append(CheckMark(key, _records_label(evidence, passed), "pass" if passed else "fail"))
        elif key == "leakage":
            # attach_finding_checks only ever writes leakage=True (a check
            # that found nothing is omitted, not written False) — this is
            # always the fail state when the key is present at all.
            marks.append(CheckMark(key, "Too good to be true", "fail"))
        elif key == "cause":
            # Never a tick or a fail — a neutral note, per the check-rules
            # table ("Cause ... neutral note, never a tick").
            marks.append(CheckMark(key, "Pattern, not proof of cause", "neutral"))
        else:
            passed = bool(value)
            pass_label, fail_label = _LABELS[key]
            marks.append(CheckMark(key, pass_label if passed else fail_label, "pass" if passed else "fail"))
    return marks
