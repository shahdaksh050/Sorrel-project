"""
Unit tests for the Swarm Dashboard construction animation and component.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from ui.components.swarm_dashboard import (
    SWARM_HTML_PATH,
    build_swarm_payload,
    get_swarm_dashboard_html,
    render_swarm_dashboard,
)


def test_swarm_html_file_exists_and_contains_expected_tokens() -> None:
    """Verify that the prototype HTML exists and contains all required agents and cards."""
    assert SWARM_HTML_PATH.is_file(), f"Prototype HTML missing at {SWARM_HTML_PATH}"

    html = get_swarm_dashboard_html(inline_vega=False)
    assert len(html) > 1000, "HTML content is unexpectedly short"

    # Verify 10 distinct agents are defined in script
    expected_agents = [
        "Orchestrator",
        "Data Profiler",
        "Metrics Bot",
        "Statistical Core",
        "Causal Auditor",
        "Vega Plotter",
        "Segment Explorer",
        "ML Tournament",
        "Action Architect",
        "Quality Verifier",
    ]
    for agent_name in expected_agents:
        assert agent_name in html, f"Agent {agent_name} not found in swarm dashboard HTML"

    # Verify key dashboard card containers exist
    expected_targets = [
        "card-header",
        "card-telemetry",
        "card-kpi-1",
        "card-kpi-2",
        "card-kpi-3",
        "card-chart-hero",
        "card-chart-segment",
        "card-models",
        "card-action-matrix",
        "seal-audit",
    ]
    for target_id in expected_targets:
        assert f'id="{target_id}"' in html, f"Target element #{target_id} not found in HTML"

    # Verify selection marquee rubberband structure exists
    assert "selection-marquee" in html
    assert "marquee-dimension" in html
    assert "marquee-handle" in html

    # Verify accessibility: prefers-reduced-motion check
    assert "prefers-reduced-motion" in html

    # Verify interactive controls exist
    assert 'id="btn-replay"' in html
    assert 'id="btn-hexagon"' in html
    assert 'id="btn-skip"' in html
    assert 'id="btn-theme"' in html

    # Verify Hexagon Core Mode elements exist
    assert 'id="hex-hub-container"' in html
    assert 'id="hex-facets-group"' in html
    assert 'id="hex-seams-group"' in html
    assert 'id="hex-math-badge"' in html
    assert "hex-payload" in html
    assert "hex-facet" in html
    assert "hex-seam-line" in html
    assert "Equi-Area Dissection" in html
    assert "runHexagonChoreography" in html

    # Verify Helper Swarm (10 Helper Sculptors for 90° Shape Rectification)
    expected_helpers = [
        "Corner Fitter",
        "Aspect Aligner",
        "Ortho Shaper",
        "Corner Puller",
        "90° Rectifier",
        "Canvas Squarer",
        "Grid Leveler",
        "Bounds Rectifier",
        "Edge Planer",
        "Seal Sculptor",
    ]
    for helper_role in expected_helpers:
        assert helper_role in html, f"Helper sculptor role '{helper_role}' not found in HTML"

    assert 'id="hex-sculpt-guides"' in html
    assert "sculptor-ruler-line" in html
    assert "sculptor-ruler-badge" in html
    assert "helper-cursor" in html
    assert "90° Shape Rectification" in html
    assert "Dual-Escort Dispatch" in html

    # Verify unified monotonous cursor colors and styling
    assert "const UNIFIED_CURSOR_COLOR = '#0f172a';" in html
    assert "const UNIFIED_HELPER_COLOR = '#0f172a';" in html
    assert "background: #334155;" in html  # Monotone helper badge
    assert "background: #94a3b8;" in html  # Monotone cursor indicator

    # Verify Day/Light mode styling & removal of muddy blur
    assert '[data-theme="day"]' in html
    assert '--hex-glow-color: transparent;' in html
    assert '[data-theme="day"] #hex-trace-path' in html
    assert '[data-theme="day"] #hex-core-glow' in html

    # Verify outside extraction staging & core dissolution
    assert "stageDist = Math.max(295, distToTarget * 0.48)" in html
    assert "coreAlpha = Math.max(0, 1 - extP * 2.5)" in html
    assert "targetRects[agId]" in html

    # Verify Vega-Lite Phase 1 integration tokens
    assert "mountVegaCharts" in html
    assert "hydrateDomFromPayload" in html
    assert ".chart-svg-box .vega-embed" in html
    assert "chart-fallback-box" in html
    assert "Visualization Render Error" in html
    assert "document.createTextNode(m.name" in html
    assert "static/vendor/vega/vega.min.js" in html


def test_render_swarm_dashboard_session_state_and_component_call() -> None:
    """Verify that render_swarm_dashboard tracks session state and invokes streamlit components.html."""
    fake_session: dict[str, bool] = {}

    with (
        patch("streamlit.session_state", fake_session),
        patch("streamlit.components.v1.html") as mock_html,
    ):
        render_swarm_dashboard(run_once=True, session_key="test_swarm_key", height=800)

        # Ensure session key was marked true
        assert fake_session.get("test_swarm_key") is True
        # Ensure components.html was invoked once with correct height
        mock_html.assert_called_once()
        args, kwargs = mock_html.call_args
        assert kwargs.get("height") == 800
        assert len(args[0]) > 1000


def test_build_swarm_payload_from_dashboard_and_report() -> None:
    """Verify payload extraction and structure from dashboard specs, report, and findings."""
    mock_dashboard: list[dict[str, Any]] = [
        {
            "chart_id": "chart_hero",
            "title": "Active Power Trend",
            "description": "Trajectory over hourly intervals",
            "spec": {"mark": "line", "data": {"values": [{"x": 1, "y": 10}]}},
        },
        {
            "chart_id": "chart_segment",
            "title": "Segment Breakdown",
            "description": "Peak vs Off-Peak load distributions",
            "spec": {"mark": "bar", "data": {"values": [{"cat": "A", "val": 20}]}},
        },
    ]
    mock_report: dict[str, Any] = {
        "source_name": "production_telemetry.csv",
        "run_id": "9042",
        "objective": "Energy Optimization Briefing",
        "lead_narrative": "Coordinated load shifting reduces evening peak active power.",
        "row_count": 52500,
        "column_count": 14,
        "quality_score": 98.7,
        "missingness": 0.0012,
        "models": [
            {
                "name": "GradientBoostingRegressor",
                "cv_mean": "0.912 ± 0.008",
                "gap": "0.024",
                "warning": "None (Healthy)",
            }
        ],
        "action_matrix": {
            "quick_wins": ["Shift laundry cycles to overnight window"],
            "strategic_pilots": ["Pilot commercial demand response at 17:00"],
            "watch_caveats": ["Grid frequency correlation lacks causal evidence"],
            "low_priority": ["Standby appliance vampire draw"],
        },
        "checksum": "8f2a...9042",
    }
    mock_findings: list[dict[str, Any]] = [
        {
            "finding_id": "f1",
            "headline": "Significant 24.5% load reduction during peak window",
            "detail": "Average peak reduction across 8,500 hourly records",
            "effect": 0.245,
            "effect_kind": "pct",
            "p_value": 0.0001,
            "dimension": "peak_window",
        }
    ]

    payload = build_swarm_payload(
        dashboard=mock_dashboard,
        report=mock_report,
        findings=mock_findings,
    )

    # Top-level sections exist
    assert "metadata" in payload
    assert "telemetry" in payload
    assert "kpis" in payload
    assert "charts" in payload
    assert "models" in payload
    assert "action_matrix" in payload
    assert "audit" in payload

    # Metadata verified
    assert payload["metadata"]["source_name"] == "production_telemetry.csv"
    assert payload["metadata"]["run_id"] == "9042"
    assert payload["metadata"]["title"] == "Energy Optimization Briefing"
    assert "load shifting" in payload["metadata"]["lead_narrative"]
    assert payload["metadata"]["headline_badge"] == "1 findings · 0 checked"

    # Telemetry verified
    assert payload["telemetry"]["records"] == "52,500"
    assert payload["telemetry"]["dimensions"] == "14 Cols"
    assert payload["telemetry"]["quality_score"] == "98.7 / 100"
    assert payload["telemetry"]["missingness"] == "0.12%"

    # KPIs verified
    assert len(payload["kpis"]) == 2
    assert payload["kpis"][0]["target"] == 24.5
    assert payload["kpis"][0]["prefix"] == "+"
    assert payload["kpis"][0]["suffix"] == "%"
    assert payload["kpis"][1]["badge"] == "p < 0.001"

    # Charts verified
    assert payload["charts"]["hero"]["title"] == "Active Power Trend"
    assert payload["charts"]["hero"]["spec"]["mark"] == "line"
    assert payload["charts"]["segment"]["title"] == "Segment Breakdown"
    assert payload["charts"]["segment"]["spec"]["mark"] == "bar"
    assert payload["charts"]["extra"] == []

    # Models and Action Matrix verified
    assert payload["models"]["has_models"] is True
    assert len(payload["models"]["model_list"]) == 1
    assert payload["action_matrix"]["has_action_matrix"] is True
    assert payload["audit"]["checksum"] == "8f2a...9042"


def test_empty_report_has_no_fabricated_audit_or_telemetry() -> None:
    with patch("streamlit.session_state", {"preview_df": [1, 2]}):
        payload = build_swarm_payload(report={"models": [{"name": "untested"}]})
    assert payload["telemetry"]["records"] == "N/A"
    assert payload["telemetry"]["quality_score"] == "Unavailable"
    assert payload["telemetry"]["missingness"] == "Unavailable"
    assert payload["kpis"] == []
    assert payload["audit"]["checksum"] is None
    assert payload["audit"]["ledger_text"] == "No finding checks recorded"
    assert payload["models"]["model_list"][0]["cv_mean"] == "Unavailable"


def test_get_swarm_dashboard_html_with_payload_injection() -> None:
    """Verify payload injection and strict escaping of script closing tags."""
    evil_title = '</script><script>alert("xss")</script>'
    payload: dict[str, Any] = {
        "metadata": {
            "source_name": "clean.csv",
            "run_id": "1111",
            "title": evil_title,
            "lead_narrative": "Narrative test",
            "headline_badge": "Verified",
        },
        "telemetry": {"records": "100", "dimensions": "2 Cols", "quality_score": "100 / 100", "missingness": "0.00%"},
        "kpis": [],
        "charts": {"hero": None, "segment": None, "extra": []},
        "models": {"has_models": False, "model_list": []},
        "action_matrix": {"has_action_matrix": False, "data": {}},
        "audit": {"claims_verified": "0 of 0", "ledger_text": "Audit", "subtext": "Sub", "checksum": "abc"},
    }

    html = get_swarm_dashboard_html(payload=payload, inline_vega=False)

    # Injected variables must be defined
    assert "window.__DSA_SWARM_PAYLOAD__ =" in html
    assert "window.__DSA_VEGA_CFG_DAY__ =" in html
    assert "window.__DSA_VEGA_CFG_NIGHT__ =" in html

    # Strict escaping check: raw </script> from payload title must NOT appear unescaped
    assert evil_title not in html
    assert "<\\/script>" in html


def test_get_swarm_dashboard_html_inlines_vega_scripts() -> None:
    """Verify inlining of vendored Vega bundles and elimination of relative script paths."""
    html_inlined = get_swarm_dashboard_html(inline_vega=True)

    # Vendored bundles should be inlined into <head>
    assert "<script>" in html_inlined
    # Relative paths must be eliminated when inlined
    assert '<script src="../../static/vendor/vega/vega.min.js"></script>' not in html_inlined
    assert '<script src="../../static/vendor/vega/vega-lite.min.js"></script>' not in html_inlined
    assert '<script src="../../static/vendor/vega/vega-embed.min.js"></script>' not in html_inlined

    # Non-inlined mode preserves standalone prototype tags
    html_non_inlined = get_swarm_dashboard_html(inline_vega=False)
    assert '<script src="../../static/vendor/vega/vega.min.js"></script>' in html_non_inlined


def test_render_swarm_dashboard_with_dashboard_param() -> None:
    """Verify render_swarm_dashboard builds payload and passes inlined HTML to components.html."""
    mock_dash = [
        {"chart_id": "c1", "title": "Live Chart 1", "description": "D1", "spec": {"mark": "bar"}},
    ]

    with patch("streamlit.components.v1.html") as mock_html:
        render_swarm_dashboard(dashboard=mock_dash, height=920, scrolling=False)
        mock_html.assert_called_once()
        args, kwargs = mock_html.call_args
        assert kwargs.get("height") == 920
        assert kwargs.get("scrolling") is False
        injected_html = args[0]
        assert "window.__DSA_SWARM_PAYLOAD__ =" in injected_html
        assert "Live Chart 1" in injected_html


def test_swarm_payload_chart_fallback_behavior() -> None:
    """Verify payload fallback handling for 0, 1, and 4+ charts."""
    # 0 Charts
    p0 = build_swarm_payload(dashboard=[])
    assert p0["charts"]["hero"] is None
    assert p0["charts"]["segment"] is None
    assert p0["charts"]["extra"] == []

    # 1 Chart
    p1 = build_swarm_payload(dashboard=[{"chart_id": "c0", "title": "Single Chart", "spec": {"mark": "point"}}])
    assert p1["charts"]["hero"] is not None
    assert p1["charts"]["hero"]["title"] == "Single Chart"
    assert p1["charts"]["segment"] is None
    assert p1["charts"]["extra"] == []

    # 4 Charts
    p4 = build_swarm_payload(
        dashboard=[{"chart_id": f"c{i}", "title": f"Chart {i}", "spec": {"mark": "line"}} for i in range(4)]
    )
    assert p4["charts"]["hero"]["title"] == "Chart 0"
    assert p4["charts"]["segment"]["title"] == "Chart 1"
    assert len(p4["charts"]["extra"]) == 2
    assert p4["charts"]["extra"][0]["title"] == "Chart 2"
    assert p4["charts"]["extra"][1]["title"] == "Chart 3"

    # Descriptive EDA mode (no models)
    p_eda = build_swarm_payload(report={"models": []})
    assert p_eda["models"]["has_models"] is False
    assert p_eda["models"]["model_list"] == []


def test_build_swarm_payload_question_coverage_and_statistical_states() -> None:
    """Verify that build_swarm_payload includes question coverage and statistical states in headline_badge."""
    mock_report = {
        "coverage": {"answered": 4, "total": 15},
    }
    mock_findings = [
        {
            "finding_id": "f_equiv",
            "headline": "Equivalence demonstrated",
            "statistical_state": "demonstrated_equivalence",
            "evidence": {"checks": {"claim_verified": True}},
        },
        {
            "finding_id": "f_diff",
            "headline": "Difference detected",
            "statistical_state": "detected_difference",
            "evidence": {"checks": {"claim_verified": True}},
        },
        {
            "finding_id": "f_inconc",
            "headline": "Inconclusive comparison",
            "statistical_state": "inconclusive",
            "evidence": {"checks": {"claim_verified": True}},
        },
    ]
    payload = build_swarm_payload(report=mock_report, findings=mock_findings)
    badge = payload["metadata"]["headline_badge"]
    assert "4/15 questions answered" in badge
    assert "1 equiv" in badge
    assert "1 diff" in badge
    assert "1 inconclusive" in badge
