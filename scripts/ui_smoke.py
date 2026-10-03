"""Browser smoke test: start the app, open it, run the bundled sample, check the result.

Needs `pip install playwright` and `playwright install chromium` (or CHROMIUM_PATH). Run from the repo root:

    python scripts/ui_smoke.py [--port 8599]

Exits 0 when the sample analysis completes and the Answers tab shows findings, with no
horizontal overflow at phone width; exits 1 otherwise. It starts and stops its own server.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import Frame, Page, sync_playwright

TIMEOUT_MS = 240_000


def _wait_for_server(url: str, seconds: float = 60.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"server did not start at {url}")


def _landing_frame(page: Page) -> Frame:
    for _ in range(120):
        for frame in page.frames:
            if frame.locator("#hero-enter-btn").count():
                return frame
        page.wait_for_timeout(500)
    raise RuntimeError("landing page did not render")


def _overflow(page: Page) -> int:
    return int(page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    ))


def run(port: int) -> list[str]:
    failures: list[str] = []
    url = f"http://localhost:{port}"
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--no-sandbox"],
            executable_path=os.environ.get("CHROMIUM_PATH") or None,
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.emulate_media(reduced_motion="reduce")
        page.goto(url)
        page.wait_for_selector(".stApp", timeout=TIMEOUT_MS)
        page.wait_for_function("window.getComputedStyle(document.documentElement).getPropertyValue('--dur-base').trim() !== ''", timeout=TIMEOUT_MS)
        dur = page.evaluate("window.getComputedStyle(document.documentElement).getPropertyValue('--dur-base').trim()")
        if dur != "0ms" and dur != "0ms !important":
            failures.append(f"reduced motion not zeroed: {dur}")

        page.emulate_media(reduced_motion="no-preference")
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)[:200]))
        page.goto(url)
        _landing_frame(page).locator("#hero-enter-btn").click()
        page.get_by_role("button", name="Try it with sample data").click(timeout=TIMEOUT_MS)
        page.wait_for_selector("[role=tab]:has-text('Answers')", timeout=TIMEOUT_MS)
        if not page.locator("text=What we found").count():
            failures.append("Answers tab has no 'What we found' section")
        for tab in ("Charts", "Details", "Downloads"):
            page.get_by_role("tab", name=tab).click()
            page.wait_for_timeout(1500)
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(1500)
        if _overflow(page) > 2:
            failures.append(f"page scrolls sideways at 390 px by {_overflow(page)} px")
        if errors:
            failures.append(f"uncaught page errors: {errors[:2]}")
        browser.close()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8599)
    args = parser.parse_args()
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(args.port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_server(f"http://localhost:{args.port}")
        failures = run(args.port)
    finally:
        server.terminate()
        server.wait(timeout=15)
    for f in failures:
        print("FAIL:", f)
    print("ui smoke:", "FAILED" if failures else "ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
