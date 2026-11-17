"""
Unit tests for the Swarm Dashboard construction animation and component.
"""
from __future__ import annotations

from unittest.mock import patch

from ui.components.swarm_dashboard import (
    SWARM_HTML_PATH,
    get_swarm_dashboard_html,
    render_swarm_dashboard,
)


def test_swarm_html_file_exists_and_contains_expected_tokens() -> None:
    """Verify that the prototype HTML exists and contains all required agents and cards."""
    assert SWARM_HTML_PATH.is_file(), f"Prototype HTML missing at {SWARM_HTML_PATH}"

    html = get_swarm_dashboard_html()
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
