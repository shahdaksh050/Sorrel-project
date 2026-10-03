"""Run the real app headlessly in THIS process and print what it rendered as one JSON line.

Not a test (no `test_` prefix, so pytest does not collect it): `tests/test_app_render.py` runs it in a
child process. `app.py` stubs `rich`, sets environment variables and starts a worker thread, so
rendering it inside the pytest process would leak into every other test, and the other tests'
imports would in turn change how the app behaves. A separate process keeps both sides clean.

No browser and no network: Streamlit's `AppTest` executes the script, and the bundled sample is
analysed without an AI summary.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

RUN_TIMEOUT_S = 150


def make() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.session_state["entered"] = True  # past the landing page
    return at


def markup(at: AppTest) -> str:
    return " ".join(str(m.value) for m in at.markdown)


def problems(at: AppTest) -> list[str]:
    return [str(e.value)[:300] for e in at.exception]


def main() -> dict[str, object]:
    out: dict[str, object] = {}

    # 1. The empty workspace.
    at = make().run()
    text = markup(at)
    out["empty"] = {
        "exceptions": problems(at),
        "welcome": "Start with the data." in text,
        "stepper": 'class="stepper"' in text,
        "sample_button": any(b.key == "btn_sample_data" for b in at.button),
    }

    # 2. A sample run: capture the page while it runs, then wait for the results.
    at = make().run()
    next(b for b in at.button if b.key == "btn_sample_data").click().run()
    running: dict[str, object] = {}
    finished = False
    deadline = time.monotonic() + RUN_TIMEOUT_S
    while time.monotonic() < deadline:
        if at.exception:
            break
        text = markup(at)
        if "Analysing" in text and not running:
            running = {
                "stepper": 'class="stepper"' in text,
                "notes_or_placeholder": 'class="step-notes"' in text or 'class="step-notes-empty"' in text,
                "exceptions": problems(at),
            }
        if "Results for" in text:
            finished = True
            break
        at.run()
        time.sleep(1)
    text = markup(at)
    stepper = text.split('class="stepper"', 1)[1].split("</ol>", 1)[0] if 'class="stepper"' in text else ""
    out["running"] = running
    out["finished"] = {
        "finished": finished,
        "exceptions": problems(at),
        "steps": len(re.findall(r'<li class="step[ "]', text)),
        "stepper_label": (re.search(r'aria-label="(Analysis steps[^"]*)"', text) or [None, ""])[1],
        "waiting_in_stepper": "○ Waiting" in stepper,
        "tabs": [t.label for t in at.tabs],
        "eyebrow": "Sorrel · Workspace" in text,
        "buttons": sorted(b.key for b in at.button if b.key in ("btn_new_analysis", "btn_toggle_hero_cinema")),
        "how_rows": 'class="how-we-got-here"' in text and 'class="how-block' in text,
        "artifacts": text.count('class="artifact'),
        "analyst_notes": "Details for analysts" in text,
        "step_notes": 'class="step-notes"' in text,
        "agent_line": "agent-line" in text,
        "agent_cards": text.count("agent-card"),
        "report_toggle": any(t.key == "show_report_preview" for t in at.toggle),
        "report_frame_before_asking": 'key="report_preview"' in text or "st-key-report_preview" in text,
    }

    # 3. A failed run, seeded directly (no API key, no network): the stepper shows the stopped step.
    at = make()
    at.session_state["analysis_error"] = "Provider said no."
    at.session_state["stage_log"] = [("1", "done", "read"), ("2", "error", "LLM unreachable")]
    at.run()
    text = markup(at)
    out["failed"] = {
        "exceptions": problems(at),
        "headline": "Could not finish" in text,
        "stopped_step": 'class="step err"' in text and "! Stopped" in text,
    }
    return out


if __name__ == "__main__":
    result = main()
    print("RESULT=" + json.dumps(result), flush=True)
    os._exit(0)  # a finished run's worker thread must not keep this process alive
