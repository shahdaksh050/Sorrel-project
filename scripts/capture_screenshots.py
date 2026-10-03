import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

TIMEOUT_MS = 120_000
OUT_DIR = Path("design-references/screenshots/step3")

def _wait_for_server(url: str, seconds: float = 60.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"server did not start at {url}")

def _landing_frame(page) -> object:
    for _ in range(120):
        for frame in page.frames:
            if frame.locator("#hero-enter-btn").count():
                return frame
        page.wait_for_timeout(500)
    raise RuntimeError("landing page did not render")

def capture():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    url = "http://localhost:8599"
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--no-sandbox"],
            executable_path=os.environ.get("CHROMIUM_PATH") or None,
        )

        for theme in ["day", "night"]:
            for width, height, device in [(1440, 900, "desktop"), (390, 844, "mobile")]:
                page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme if theme == "dark" else "light")
                page.goto(url)

                # We can't strictly force the theme unless we use the UI selectbox, but we can set localStorage or select it
                page.wait_for_selector(".stApp", timeout=TIMEOUT_MS)

                # 1. Landing state
                page.screenshot(path=str(OUT_DIR / f"01_landing_{theme}_{device}.png"))

                # Click enter
                _landing_frame(page).locator("#hero-enter-btn").click()
                page.wait_for_timeout(1000)

                # Switch theme if needed using the selectbox in the sidebar
                # The landing page is day/night based on system preference if we used color_scheme.

                # 2. Empty state
                page.screenshot(path=str(OUT_DIR / f"02_empty_{theme}_{device}.png"))

                # 3. Uploaded dataset state
                file_input = page.locator("input[type='file']").first
                file_input.set_input_files("test_csv.csv")
                page.wait_for_selector("text=Related tables", timeout=TIMEOUT_MS)
                page.wait_for_timeout(2000)
                page.screenshot(path=str(OUT_DIR / f"03_uploaded_{theme}_{device}.png"))

                # 4. Mid-run state
                page.locator("textarea").first.fill("Run an analysis")
                page.get_by_placeholder("sk-...").fill("sk-dummy-key-for-test")
                page.get_by_role("button", name="Run Analysis").click(force=True)
                page.wait_for_timeout(2000) # wait to capture mid-run
                page.screenshot(path=str(OUT_DIR / f"04_midrun_{theme}_{device}.png"))

                # 5. Completed state
                # Wait for the "Found so far" to disappear and the Tabs to appear
                # The completed state shows Tabs with role="tablist"
                page.wait_for_selector(".stTabs", timeout=120_000)
                page.wait_for_timeout(2000) # Give it a moment to render charts
                page.screenshot(path=str(OUT_DIR / f"05_completed_{theme}_{device}.png"), full_page=True)

                page.close()

        browser.close()

if __name__ == "__main__":
    env = os.environ.copy()
    env["LOCAL_ONLY"] = "1"
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8599"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env
    )
    try:
        _wait_for_server("http://localhost:8599")
        capture()
    finally:
        server.terminate()
        server.wait(timeout=15)
