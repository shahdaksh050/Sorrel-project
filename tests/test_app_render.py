"""The real app, rendered end to end with Streamlit's own test harness (`AppTest`).

No browser and no network: the bundled sample is analysed without an AI summary, in a few seconds. This
catches what unit tests on single functions cannot: an exception in the page's layout code, a slot that
is never filled, a tab that fails to render. It cannot judge how the page looks; that needs eyes.

The app runs in a child process (`tests/app_render_driver.py`). It stubs `rich`, sets environment
variables and starts a worker thread, so rendering it inside the pytest process leaks into other
tests (the controller tests failed after it), and other tests' imports change how it behaves.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests" / "app_render_driver.py"


@pytest.fixture(scope="module")
def rendered() -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, str(DRIVER)], cwd=ROOT, capture_output=True, text=True, timeout=280, check=False
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT=")]
    assert lines, f"the driver printed no result (exit {done.returncode}):\n{done.stderr[-1500:]}"
    result: dict[str, Any] = json.loads(lines[-1][len("RESULT=") :])
    return result


def test_the_empty_workspace_renders_without_error(rendered: dict[str, Any]) -> None:
    empty = rendered["empty"]
    assert empty["exceptions"] == []
    assert empty["welcome"] and empty["sample_button"]
    assert not empty["stepper"]  # no run, no steps


def test_a_running_page_shows_the_stepper_beside_the_plate(rendered: dict[str, Any]) -> None:
    running = rendered["running"]
    assert running, "the page was never seen in its running state"
    assert running["exceptions"] == []
    assert running["stepper"]


def test_a_finished_run_shows_the_header_band_the_stepper_and_all_four_tabs(rendered: dict[str, Any]) -> None:
    done = rendered["finished"]
    assert done["finished"], "the sample run did not reach its results page"
    assert done["exceptions"] == []
    assert done["steps"] == 7
    # A step the run did not need (here "Solving the Tricky Parts") is skipped, not done, so the count can be 6.
    assert re.fullmatch(r"Analysis steps, [67] of 7 done", done["stepper_label"])
    assert not done["waiting_in_stepper"]
    assert done["tabs"] == ["Answers", "Charts", "Details", "Downloads"]
    assert done["eyebrow"]
    assert done["buttons"] == ["btn_new_analysis", "btn_toggle_hero_cinema"]


def test_a_finished_run_explains_itself_in_the_details_and_offers_its_files(rendered: dict[str, Any]) -> None:
    done = rendered["finished"]
    assert done["how_rows"]  # audit rows, not a grid
    assert done["artifacts"] >= 3  # the shelf: report, data, models
    assert done["analyst_notes"]


def test_the_home_page_has_no_how_it_works_panel_and_the_team_lives_in_details(rendered: dict[str, Any]) -> None:
    done = rendered["finished"]
    assert not done["how_panel"]  # no expander and no "The Team at Work" under the steps
    assert done["step_details"] >= 4  # what each step reported (rows read, tools run...) is in its own row
    # The compact team cards still exist, in the Details tab's step-by-step record.
    assert done["agent_line"] and done["agent_cards"] >= 8


def test_details_leads_with_the_analyses_that_ran_for_this_data(rendered: dict[str, Any]) -> None:
    assert rendered["finished"]["details_before_file"]


def test_the_report_preview_is_not_sent_until_it_is_asked_for(rendered: dict[str, Any]) -> None:
    done = rendered["finished"]
    assert done["report_toggle"]
    assert not done["report_frame_before_asking"]


def test_a_failed_run_shows_the_stopped_step_in_amber_without_error(rendered: dict[str, Any]) -> None:
    failed = rendered["failed"]
    assert failed["exceptions"] == []
    assert failed["headline"] and failed["stopped_step"]
