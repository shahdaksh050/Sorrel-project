"""
"Found so far, may change": the live findings list shown while a run is going.

Pure HTML builder: no Streamlit, no session state, every dynamic string escaped,
classes only. These findings have not been through the checks, the run-level
correction or the audits, so the block says so and never renders a check mark.
"""
from __future__ import annotations

import html

from src.core.run_view import ProvisionalFinding

#: How many of the latest findings to list; the rest are counted, not shown.
MAX_SHOWN = 8


def _kind_label(kind: str) -> str:
    text = kind.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "Finding"


def build_provisional_html(findings: tuple[ProvisionalFinding, ...], seen_provs: set[str], *, shown: int = MAX_SHOWN) -> str:
    """The provisional block, or `""` when nothing has been found yet."""
    if not findings:
        return ""
    latest = findings[-shown:]
    total = len(findings)
    rows = []
    for i, f in enumerate(latest):
        is_new = False
        if f.finding_id not in seen_provs:
            is_new = True
            seen_provs.add(f.finding_id)
        
        # In provisional findings, we don't know risk flags yet, so just standard cards.
        classes = ["bento-card", "prov-item"]
        if is_new:
            classes.append("is-new")
        if i == 0:
            classes.append("full-width")
            
        rows.append(
            f'<div class="{" ".join(classes)}">'
            f'<span class="prov-kind">{html.escape(_kind_label(f.kind))}</span>'
            f'<div class="finding-headline">{html.escape(f.headline)}</div>'
            f'</div>'
        )
    rows_html = "".join(rows)
    count = (
        f'<p class="prov-count">Showing the latest {len(latest)} of {total}.</p>'
        if total > len(latest)
        else ""
    )
    return (
        '<div class="prov">'
        '<h4 class="prov-h">Found so far, may change</h4>'
        '<p class="prov-note">These have not been checked yet. They can change or '
        "disappear when the checks run.</p>"
        f'<div class="bento-grid">{rows_html}</div>{count}</div>'
    )
