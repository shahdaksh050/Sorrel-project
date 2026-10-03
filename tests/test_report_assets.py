"""The shared HTML report opens offline: inline fonts and Vega, with CDN fallbacks and data tables."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.core import report_assets
from src.core.html_report import build_html_report

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>'

INSIGHTS: dict[str, Any] = {"reasoning": "r", "insights": ["i"], "recommendations": ["do"]}


def _chart(values: list[dict[str, Any]] | None = None, **spec_extra: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"mark": "bar", "encoding": {}}
    if values is not None:
        spec["data"] = {"values": values}
    spec.update(spec_extra)
    return {"chart_id": "c", "title": "Chart title", "description": "d", "spec": spec}


def _report(charts: list[dict[str, Any]]) -> str:
    return build_html_report("ds", INSIGHTS, [], charts)


# ── assets present ───────────────────────────────────────────────────────────


def test_report_with_static_present_makes_no_remote_request() -> None:
    doc = _report([_chart([{"x": 1, "y": 2.5}])])
    assert "fonts.googleapis.com" not in doc and "fonts.gstatic.com" not in doc
    assert "cdn.jsdelivr.net" not in doc
    assert not re.search(r"<script[^>]+src=", doc)
    assert not re.search(r"<link[^>]+href=\"https?:", doc)
    assert "data:font/woff2;base64," in doc
    assert "vegaEmbed" in doc


def test_inline_vega_scripts_load_in_dependency_order() -> None:
    """vega-embed needs vega and vega-lite on the page first; the chart script comes last."""
    doc = _report([_chart([{"x": 1}])])
    scripts = re.findall(r"<script>(.*?)</script>", doc, re.DOTALL)
    assert len(scripts) == 4
    assert ".vega=" in scripts[0][:700]
    assert ".vegaLite=" in scripts[1][:700]
    assert ".vegaEmbed=" in scripts[2][:700]
    assert scripts[3].startswith("const SPECS")


def _rules(doc: str, family: str) -> list[str]:
    return re.findall(rf"@font-face\{{font-family:'{family}'[^}}]*\}}", doc)


def test_identical_font_files_are_embedded_once_with_a_weight_range() -> None:
    """Sorrel's faces are variable fonts: one file, one rule, the whole weight range."""
    doc = _report([_chart([{"x": 1}])])
    geist, mono, newsreader = _rules(doc, "Geist"), _rules(doc, "Geist Mono"), _rules(doc, "Newsreader")
    assert len(geist) == 1 and "font-weight:100 900" in geist[0]
    assert len(mono) == 1 and "font-weight:100 900" in mono[0]
    assert len(newsreader) == 2 and all("font-weight:200 800" in rule for rule in newsreader)
    assert sorted(re.search(r"font-style:(\w+)", rule).group(1) for rule in newsreader) == [  # type: ignore[union-attr]
        "italic",
        "normal",
    ]
    assert all("unicode-range:" in rule for rule in geist + mono + newsreader)


def test_only_the_sorrel_families_are_embedded() -> None:
    """The stylesheet still declares the old families until the cleanup; none may reach the report."""
    fonts = report_assets.report_assets(need_vega=False).fonts_html
    families = set(re.findall(r"@font-face\{font-family:'([^']+)'", fonts))
    assert families == {"Geist", "Geist Mono", "Newsreader"}
    for old in ("Baloo", "Mukta", "Bricolage", "Public Sans", "IBM Plex"):
        assert old not in fonts


def test_faces_declared_per_weight_over_one_file_still_collapse_to_one_ranged_rule(tmp_path: Path) -> None:
    """The dedupe path the real (variable) files no longer exercise: one file declared at several weights."""
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    (fonts / "geist-latin.woff2").write_bytes(b"wOF2-one-file")
    (fonts / "newsreader-latin.woff2").write_bytes(b"wOF2-another")
    faces = "".join(
        f"@font-face {{ font-family: 'Geist'; font-style: normal; font-weight: {w}; "
        "src: url('./geist-latin.woff2') format('woff2'); unicode-range: U+0000-00FF; }\n"
        for w in (400, 600, 700)
    ) + (
        "@font-face { font-family: 'Newsreader'; font-style: normal; font-weight: 400; "
        "src: url('./newsreader-latin.woff2') format('woff2'); unicode-range: U+0000-00FF; }\n"
        # not a Sorrel family and a file that does not exist: skipped before it is read
        "@font-face { font-family: 'Baloo 2'; font-weight: 500; "
        "src: url('./baloo-2-500-latin.woff2') format('woff2'); }\n"
    )
    (fonts / "ledger-fonts.css").write_text(faces, encoding="utf-8")
    assets = report_assets.report_assets(need_vega=False, root=tmp_path)
    assert assets.offline
    geist = _rules(assets.fonts_html, "Geist")
    assert len(geist) == 1 and "font-weight:400 700" in geist[0]
    newsreader = _rules(assets.fonts_html, "Newsreader")
    assert len(newsreader) == 1 and "font-weight:400;" in newsreader[0]
    assert "Baloo" not in assets.fonts_html


def test_the_embedded_fonts_are_the_ones_the_report_css_asks_for() -> None:
    doc = _report([_chart([{"x": 1}])])
    css = doc[doc.index("<style>", doc.index("</style>")) :]  # the report's own stylesheet, after the font faces
    assert "'Geist'" in css and "'Newsreader'" in css
    assert "Baloo" not in css and "Mukta" not in css


def test_report_without_charts_embeds_fonts_but_no_vega() -> None:
    doc = _report([])
    with_charts = _report([_chart([{"x": 1}])])
    assert "data:font/woff2;base64," in doc
    assert "vegaEmbed" not in doc and "cdn.jsdelivr.net" not in doc
    assert len(with_charts) - len(doc) > 500_000  # the Vega bundles are the bulk


# ── assets absent: controlled fallback ───────────────────────────────────────


def test_missing_static_falls_back_to_the_cdn_tags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(report_assets, "ASSET_ROOT", tmp_path)
    doc = _report([_chart([{"x": 1}])])
    assert "https://fonts.googleapis.com/css2" in doc
    assert "https://cdn.jsdelivr.net/npm/vega@5" in doc
    assert "https://cdn.jsdelivr.net/npm/vega-embed@6" in doc
    assert "data:font/woff2" not in doc
    assert not report_assets.report_assets(need_vega=True, root=tmp_path).offline


def test_each_asset_falls_back_on_its_own(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shutil.copytree(STATIC / "fonts", tmp_path / "fonts")  # fonts only, no vendored Vega
    monkeypatch.setattr(report_assets, "ASSET_ROOT", tmp_path)
    doc = _report([_chart([{"x": 1}])])
    assert "data:font/woff2;base64," in doc and "fonts.googleapis.com" not in doc
    assert "https://cdn.jsdelivr.net/npm/vega@5" in doc


def test_a_missing_font_file_uses_the_cdn_not_half_the_fonts(tmp_path: Path) -> None:
    shutil.copytree(STATIC / "fonts", tmp_path / "fonts")
    (tmp_path / "fonts" / "newsreader-latin-wght-italic.woff2").unlink()
    assets = report_assets.report_assets(need_vega=False, root=tmp_path)
    assert assets.fonts_html == report_assets.FONTS_CDN and not assets.offline


def test_the_cdn_fallback_asks_for_the_sorrel_families_only() -> None:
    cdn = report_assets.FONTS_CDN
    assert "family=Geist:" in cdn and "family=Geist+Mono:" in cdn and "family=Newsreader:" in cdn
    assert "Baloo" not in cdn and "Mukta" not in cdn


def test_a_literal_closing_script_tag_in_a_bundle_cannot_end_the_tag_early(tmp_path: Path) -> None:
    vega = tmp_path / "vendor" / "vega"
    vega.mkdir(parents=True)
    for name in ("vega.min.js", "vega-lite.min.js", "vega-embed.min.js"):
        (vega / name).write_text('var a = "</script><b>"; var b = "</SCRIPT>";', encoding="utf-8")
    html = report_assets.report_assets(need_vega=True, root=tmp_path).vega_html
    assert html.count("</script>") == 3  # only our own three closers
    assert "<\\/script" in html and "<\\/SCRIPT" in html


# ── data tables behind charts ────────────────────────────────────────────────


def test_every_chart_with_inline_data_gets_a_server_rendered_table() -> None:
    doc = _report([_chart([{"group": "a", "n": 3, "rate": 0.123456}, {"group": "b", "n": 5, "rate": 0.5}])])
    holder = re.search(r'<div class="vega-holder" id="chart_0">(.*?)</div></div>', doc, re.DOTALL)
    assert holder is not None
    table = holder.group(1)
    assert 'class="chart-data"' in table
    assert "<th>group</th>" in table and "<td>0.1235</td>" in table
    assert "Data behind this chart (2 rows)" in table


def test_layered_specs_use_their_first_layer_data() -> None:
    chart = _chart(None, layer=[{"data": {"values": [{"k": 1}]}, "mark": "line"}])
    assert 'class="chart-data"' in _report([chart])


def test_no_table_is_invented_when_the_spec_has_no_inline_data() -> None:
    doc = _report([_chart(None, data={"url": "data.csv"})])
    assert 'class="chart-data"' not in doc


def test_long_tables_are_capped_and_say_so() -> None:
    values = [{"i": i} for i in range(55)]
    doc = _report([_chart(values)])
    assert "first 20 of 55 rows" in doc
    assert doc.count("<td>") >= 20 and "<td>54</td>" not in doc


def test_hostile_chart_data_is_escaped_in_the_table() -> None:
    doc = _report([_chart([{"label": HOSTILE, HOSTILE: 1}])])
    match = re.search(r'<div class="vega-holder" id="chart_0">(.*?)</div></div>', doc, re.DOTALL)
    assert match is not None
    holder = match.group(1)
    assert "<script" not in holder and "<img" not in holder
    assert "&lt;script&gt;" in holder


def test_the_chart_script_replaces_the_table_on_success_and_keeps_it_on_failure() -> None:
    doc = _report([_chart([{"x": 1}])])
    script = doc[doc.index("const SPECS") :]
    assert "typeof vegaEmbed !== 'undefined'" in script  # no Vega: the table just stays
    assert "chart-data-details" in script and "Show the data behind this chart" in script
    assert ".catch(() => { target.remove(); })" in script  # a failed chart leaves the table visible


def test_print_hides_the_collapsed_data_details_and_uses_tokens() -> None:
    doc = _report([_chart([{"x": 1}])])
    assert ".chart-data-details { display: none; }" in doc.split("@media print")[1]
    css = doc[doc.index(".chart-data {") : doc.index("@media print")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css)


def test_the_embedded_chart_node_is_full_width() -> None:
    """Regression: vega-embed makes its target inline-block, which collapses a
    width:"container" chart to 0 pixels unless the node is explicitly full width.
    Found by opening a real report in a browser; unit tests cannot see layout."""
    doc = _report([_chart([{"x": 1}])])
    assert ".vega-holder .vega-embed { width: 100%; }" in doc
