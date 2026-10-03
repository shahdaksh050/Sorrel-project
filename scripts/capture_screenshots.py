"""Capture the UI state matrix (landing, empty, uploaded, running, completed) in Day and Night.

Night is chosen with the app's own top-bar Day/Night control, never with ``color_scheme``: the app's
theme is session state, so the browser preference does not change it. Every shot is recorded in
``manifest.json`` with its theme, viewport, file, objective and whether WebGL initialised, so a
screenshot can be judged by what it shows rather than by its filename.

Run: ``python scripts/capture_screenshots.py`` (needs Playwright and a Chromium).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

TIMEOUT_MS = 120_000
OUT_DIR = Path("design-references/screenshots/step3")
DATASET = "test_csv.csv"
OBJECTIVE = "Run an analysis"
URL = "http://localhost:8599"
VIEWPORTS = [(1440, 900, "desktop"), (390, 844, "mobile")]


def _wait_for_server(url: str, seconds: float = 60.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"server did not start at {url}")


def _landing_frame(page: Page) -> Any:
    for _ in range(120):
        for frame in page.frames:
            if frame.locator("#hero-enter-btn").count():
                return frame
        page.wait_for_timeout(500)
    raise RuntimeError("landing page did not render")


def _webgl_ok(page: Page) -> bool:
    return bool(page.evaluate(
        "() => { const c = document.createElement('canvas');"
        " return !!(c.getContext('webgl2') || c.getContext('webgl')); }"
    ))


def _set_theme(page: Page, theme: str) -> None:
    """Click the Day/Night control in the top bar and wait for the rerun to apply."""
    label = "Night" if theme == "night" else "Day"
    page.get_by_role("button", name=label, exact=True).first.click()
    page.wait_for_timeout(1500)


def capture() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--no-sandbox"],
            executable_path=os.environ.get("CHROMIUM_PATH") or None,
        )
        for theme in ("day", "night"):
            for width, height, device in VIEWPORTS:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(URL)
                page.wait_for_selector(".stApp", timeout=TIMEOUT_MS)

                def shot(state: str, full_page: bool = False, *, _page: Page = page,
                         _theme: str = theme, _device: str = device, _w: int = width, _h: int = height) -> None:
                    name = f"{state}_{_theme}_{_device}.png"
                    _page.screenshot(path=str(OUT_DIR / name), full_page=full_page)
                    manifest.append({
                        "file": name, "state": state, "theme": _theme, "viewport": [_w, _h],
                        "dataset": DATASET if state != "01_landing" and state != "02_empty" else None,
                        "objective": OBJECTIVE if state in ("04_running", "05_completed") else None,
                        "webgl": _webgl_ok(_page),
                    })

                shot("01_landing")
                _landing_frame(page).locator("#hero-enter-btn").click()
                page.wait_for_selector("text=Add your file", timeout=TIMEOUT_MS)
                if theme == "night":
                    _set_theme(page, "night")
                shot("02_empty")

                page.locator("input[type='file']").first.set_input_files(DATASET)
                page.wait_for_selector("text=Dataset preview", timeout=TIMEOUT_MS)
                page.wait_for_timeout(1500)
                shot("03_uploaded")

                # A real run needs no key: choose the deterministic route.
                page.locator("textarea").first.fill(OBJECTIVE)
                page.get_by_role("button", name="Without an AI summary", exact=True).first.click()
                page.wait_for_timeout(1000)
                page.get_by_role("button", name="Run analysis").click(force=True)
                page.wait_for_timeout(2500)
                shot("04_running")

                page.wait_for_selector(".stTabs", timeout=TIMEOUT_MS)
                page.wait_for_timeout(2000)  # let charts render
                shot("05_completed", full_page=True)
                page.close()
        browser.close()
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    env = os.environ.copy()
    env["LOCAL_ONLY"] = "1"
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8599"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        _wait_for_server(URL)
        capture()
    finally:
        server.terminate()
        server.wait(timeout=15)
