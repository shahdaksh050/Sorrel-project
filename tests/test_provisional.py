"""ui/components/provisional.py: the live "Found so far, may change" block."""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from ui.components.provisional import MAX_SHOWN, build_provisional_html

from src.core.audited_entry import GLYPH
from src.core.run_view import ProvisionalFinding, build_run_view

HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>'
ROOT = Path(__file__).resolve().parents[1]


def _items(n: int) -> tuple[ProvisionalFinding, ...]:
    return tuple(ProvisionalFinding(f"f{i}", "segment_lift", f"Group {i} differs") for i in range(n))


def test_nothing_found_renders_nothing() -> None:
    assert build_provisional_html(()) == ""


def test_block_is_labelled_provisional_and_lists_findings() -> None:
    out = build_provisional_html(_items(2))
    assert "Found so far, may change" in out
    assert "have not been checked yet" in out
    assert "Group 0 differs" in out and "Group 1 differs" in out
    assert "Segment lift" in out  # kind shown in plain words, not snake_case
    assert "segment_lift" not in out


def test_no_provisional_entry_carries_a_check_mark() -> None:
    out = build_provisional_html(_items(3))
    for glyph in GLYPH.values():
        assert glyph not in out
    for word in ("check-row", "held up", "Not luck", "class=\"check"):
        assert word not in out
    # The type itself has nowhere to put evidence or checks.
    assert [f.name for f in dataclasses.fields(ProvisionalFinding)] == ["finding_id", "kind", "headline"]


def test_hostile_text_is_escaped() -> None:
    items = (ProvisionalFinding("x", HOSTILE, HOSTILE),)
    out = build_provisional_html(items)
    assert "<script" not in out and "<img" not in out
    assert "&lt;script&gt;" in out


def test_only_the_latest_are_listed_and_the_rest_are_counted() -> None:
    out = build_provisional_html(_items(MAX_SHOWN + 5))
    assert out.count("<li>") == MAX_SHOWN
    assert f"Showing the latest {MAX_SHOWN} of {MAX_SHOWN + 5}." in out
    assert "Group 0 differs" not in out  # oldest dropped
    assert f"Group {MAX_SHOWN + 4} differs" in out


def test_no_count_line_when_everything_is_shown() -> None:
    assert "Showing the latest" not in build_provisional_html(_items(2))


def test_no_inline_style_and_single_line_markup() -> None:
    out = build_provisional_html(_items(3))
    assert "style=" not in out
    assert "\n" not in out


def test_every_class_used_is_styled_with_tokens_only() -> None:
    styles = (ROOT / "ui" / "styles.py").read_text(encoding="utf-8")
    names = {c for cls in re.findall(r'class="([^"]+)"', build_provisional_html(_items(MAX_SHOWN + 1))) for c in cls.split()}
    for name in sorted(names):
        assert f".{name}" in styles, f"missing CSS for .{name}"
    start = styles.index('/* ── "Found so far, may change"')
    block = styles[start : styles.index("</style>", start)]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)


def test_a_finished_runs_view_has_no_provisional_findings() -> None:
    view = build_run_view(
        {"findings": [{"finding_id": "a", "layer": "exec", "kind": "driver", "headline": "H"}]},
        {},
        objective="q",
        is_sample=False,
    )
    assert view.provisional_findings == ()
