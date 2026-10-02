"""
"How we got here": the Details-tab audit trail, as one pure HTML string.

Reads a `HowWeGotHere` (src/core/run_view.py) and returns markup for
`st.markdown(..., unsafe_allow_html=True)`. No Streamlit and no session state,
so it is testable on its own. Every dynamic string goes through `html.escape`;
styling is by class only (rules live in `ui/styles.py`).

A section with nothing to say is omitted, never rendered as a placeholder, and
the function returns `""` when no section applies.
"""
from __future__ import annotations

import html

from src.core.run_view import HowWeGotHere

_MODE_LABELS: dict[str, str] = {
    "describe": "Describe the data. No prediction was attempted.",
    "model": "Predict a target column.",
}

_STATUS_LABELS: dict[str, str] = {
    "supported": "Supported",
    "refuted": "Refuted",
    "inconclusive": "Inconclusive",
}


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _list(items: tuple[str, ...], css: str = "how-list") -> str:
    return f'<ul class="{css}">' + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"


def _more(total: int, shown: int) -> str:
    extra = total - shown
    return f'<p class="how-more">and {extra} more.</p>' if extra > 0 else ""


def _block(title: str, explainer: str, body: str) -> str:
    return (
        '<section class="how-block">'
        f'<h4 class="how-h">{_e(title)}</h4>'
        f'<p class="how-note">{_e(explainer)}</p>'
        f"{body}</section>"
    )


def _decided(how: HowWeGotHere) -> str:
    d = how.decision
    if d is None:
        return ""
    label = _MODE_LABELS.get(d.mode, d.mode)
    body = f'<p class="how-lead">{_e(label)}</p>'
    if d.rationale:
        body += f'<p class="how-text">{_e(d.rationale)}</p>'
    if d.rejected:
        body += '<p class="how-sub">Options we turned down</p>' + _list(d.rejected)
    return _block(
        "What we decided",
        "Whether to describe the data or predict something, and why.",
        body,
    )


def _changed(how: HowWeGotHere) -> str:
    if not how.fallbacks:
        return ""
    body = _list(how.fallbacks) + _more(how.fallbacks_total, len(how.fallbacks))
    return _block(
        "What changed along the way",
        "Where a planned step could not run and something simpler was used instead.",
        body,
    )


def _untraced(how: HowWeGotHere) -> str:
    if how.untraced_total <= 0:
        return ""
    n = how.untraced_total
    body = (
        f'<p class="how-lead">{n} {_plural(n, "number", "numbers")} could not be traced '
        "to a computed result.</p>"
        + _list(how.untraced_numbers)
        + _more(n, len(how.untraced_numbers))
    )
    return _block(
        "Numbers we could not trace",
        "These are marked in the written summary. Do not rely on them without a check.",
        body,
    )


def _asked_for(how: HowWeGotHere) -> str:
    dl = how.deliverables
    if dl is None:
        return ""
    body = ""
    for label, items in (
        ("Delivered", dl.delivered),
        ("Missing", dl.missing),
        ("Repaired", dl.repaired),
    ):
        if items:
            body += f'<p class="how-sub">{label}</p>' + _list(items)
    return _block(
        "What you asked for",
        "The outputs your question called for, checked against what was produced.",
        body,
    )


def _ideas(how: HowWeGotHere) -> str:
    if not how.hypotheses:
        return ""
    counts = ", ".join(
        f"{n} {_STATUS_LABELS[s].lower()}" for s, n in how.hypothesis_counts if s in _STATUS_LABELS
    )
    rows = "".join(
        f'<li><span class="how-tag {_e(h.status)}">{_e(_STATUS_LABELS.get(h.status, h.status))}'
        f"</span> {_e(h.statement)}</li>"
        for h in how.hypotheses
    )
    body = (f'<p class="how-lead">{_e(counts)}.</p>' if counts else "") + (
        f'<ul class="how-list">{rows}</ul>'
    )
    return _block(
        "Ideas we tested",
        "Explanations the analysis tried to confirm or rule out.",
        body,
    )


def _cost(how: HowWeGotHere) -> str:
    u = how.usage
    if u is not None:
        calls = f"{u.calls} AI {_plural(u.calls, 'call', 'calls')}"
        bits = [calls, f"{u.tokens:,} tokens"]
        # "&#36;" not "$": Streamlit's markdown would read a pair of dollar signs as math.
        bits.append(
            f"{'estimated cost about' if u.is_estimate else 'cost'} &#36;{u.cost_usd:,.4f}"
        )
        body = f'<p class="how-lead">{", ".join(bits)}.</p>'
        if u.is_estimate:
            body += (
                '<p class="how-text">The cost is an estimate from approximate '
                "per-token rates, not a bill.</p>"
            )
    elif how.no_ai:
        body = '<p class="how-lead">No AI was used in this run.</p>'
    else:
        return ""
    return _block("What it cost", "How much AI work went into this answer.", body)


def build_how_html(how: HowWeGotHere) -> str:
    """The "How we got here" section, or `""` when no part of it applies."""
    blocks = "".join(
        part(how) for part in (_decided, _changed, _untraced, _asked_for, _ideas, _cost)
    )
    if not blocks:
        return ""
    return (
        '<div class="how-we-got-here">'
        '<h3 class="how-title">How we got here</h3>'
        '<p class="how-lede">What the analysis decided, what it had to work around, '
        "and what it could not verify.</p>"
        f'<div class="how-grid">{blocks}</div>'
        "</div>"
    )
