"""The real app, rendered end to end with Streamlit's own test harness (`AppTest`).

No browser and no network: the bundled sample is analysed without an AI summary, in a few seconds.
This catches what unit tests on single functions cannot: an exception in the page's layout code, a slot
that is never filled, a tab that fails to render. It cannot judge how the page looks; that needs eyes.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
RUN_TIMEOUT_S = 150


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["entered"] = True  # past the landing page
    return at


def _markup(at: AppTest) -> str:
    return " ".join(str(m.value) for m in at.markdown)


def test_the_empty_workspace_renders_without_error() -> None:
    at = _app().run()
    assert not at.exception
    text = _markup(at)
    assert "Start with the data." in text
    assert 'class="stepper"' not in text  # no run, no steps
    assert any(b.key == "btn_sample_data" for b in at.button)


@pytest.fixture(scope="module")
def finished_run() -> AppTest:
    at = _app().run()
    next(b for b in at.button if b.key == "btn_sample_data").click().run()
    assert not at.exception
    deadline = time.monotonic() + RUN_TIMEOUT_S
    while time.monotonic() < deadline:
        at.run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        if "Results for" in _markup(at):
            return at
        time.sleep(2)
    pytest.fail(f"the sample run did not finish within {RUN_TIMEOUT_S}s")


def test_a_finished_run_shows_the_header_band_the_stepper_and_all_four_tabs(finished_run: AppTest) -> None:
    text = _markup(finished_run)
    assert len(re.findall(r'<li class="step[ "]', text)) == 7
    # A step the run did not need (here "Solving the Tricky Parts") is skipped, not done, so the count can be 6.
    assert re.search(r"Analysis steps, [67] of 7 done", text)
    assert "○ Waiting" not in text.split('class="stepper"', 1)[1].split("</ol>", 1)[0]
    assert [t.label for t in finished_run.tabs] == ["Answers", "Charts", "Details", "Downloads"]
    assert "Sorrel · Workspace" in text
    keys = {b.key for b in finished_run.button}
    assert {"btn_new_analysis", "btn_toggle_hero_cinema"} <= keys


def test_a_finished_run_explains_itself_in_the_details_and_offers_its_files(finished_run: AppTest) -> None:
    text = _markup(finished_run)
    assert 'class="how-we-got-here"' in text and 'class="how-block' in text  # audit rows, not a grid
    assert text.count('class="artifact') >= 3  # the shelf: report, data, models
    assert "Details for analysts" in text


def test_the_closed_how_it_works_panel_lists_what_each_step_found_and_a_compact_team(finished_run: AppTest) -> None:
    text = _markup(finished_run)
    assert 'class="step-notes"' in text
    assert "agent-line" in text and text.count("agent-card") >= 8
