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


def test_identical_font_files_are_embedded_once_with_a_weight_range() -> None:
    doc = _report([_chart([{"x": 1}])])
    baloo = re.findall(r"@font-face\{font-family:'Baloo 2'[^}]*\}", doc)
    mukta = re.findall(r"@font-face\{font-family:'Mukta'[^}]*\}", doc)
    assert len(baloo) == 1 and "font-weight:500 800" in baloo[0]
    assert len(mukta) == 4
    assert all("unicode-range:" in rule for rule in baloo + mukta)


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
    (tmp_path / "fonts" / "mukta-600-latin.woff2").unlink()
    assets = report_assets.report_assets(need_vega=False, root=tmp_path)
    assert assets.fonts_html == report_assets.FONTS_CDN and not assets.offline


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
