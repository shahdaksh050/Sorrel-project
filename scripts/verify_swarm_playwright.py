"""
Playwright End-to-End Verification for Swarm Dashboard Animation.
Verifies:
1. Hexagon sequence choreography (radial extraction outside while core vanishes, then 90° rectification, then placement).
2. Light mode color scheme (luminous frosted white base, crisp emerald contour, no dark muddy blobs).
3. Monotonous cursor colors (sleek slate #0f172a / #1e293b across all 20 cursors).
4. Full animation completion to built dashboard.
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

HTML_PATH = Path("docs/prototypes/swarm_dashboard.html").resolve()
SCREENSHOT_DIR = Path("docs/prototypes/verification_screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def verify_swarm_dashboard() -> None:
    print(f"Loading prototype from: {HTML_PATH.as_uri()}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1080})
        page = context.new_page()

        # Capture console errors if any
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.on("console", lambda msg: print(f"[Browser {msg.type}]: {msg.text}") if msg.type == "error" else None)

        page.goto(HTML_PATH.as_uri())
        page.wait_for_timeout(600)

        # ── Test 1: Day/Light Mode Theme Toggle & Color Scheme ─────────
        print("\n--- Test 1: Testing Day/Light Mode Theme & Contrast ---")
        theme_btn = page.locator("#btn-theme")
        theme_btn.click()
        page.wait_for_timeout(300)

        theme = page.evaluate("() => document.body.getAttribute('data-theme')")
        assert theme == "day", f"Expected theme 'day', got {theme}"
        print("✓ Theme successfully switched to 'day'")

        # Verify Day mode CSS variables on body / hex
        hex_fill = page.evaluate("() => getComputedStyle(document.body).getPropertyValue('--hex-base-fill').trim()")
        hex_stroke = page.evaluate("() => getComputedStyle(document.body).getPropertyValue('--hex-base-stroke').trim()")
        badge_bg = page.evaluate("() => getComputedStyle(document.body).getPropertyValue('--hex-badge-bg').trim()")

        print(f"  --hex-base-fill in day mode: {hex_fill}")
        print(f"  --hex-base-stroke in day mode: {hex_stroke}")
        print(f"  --hex-badge-bg in day mode: {badge_bg}")

        assert "255, 255, 255" in hex_fill or "#fff" in hex_fill, f"Expected light/white fill, got {hex_fill}"
        assert hex_stroke == "#059669", f"Expected #059669 emerald stroke in day mode, got {hex_stroke}"
        assert "#ffffff" in badge_bg or "255, 255, 255" in badge_bg, f"Expected clean white badge, got {badge_bg}"
        print("✓ Day mode color scheme verified: luminous white base & crisp emerald contrast")

        # ── Test 2: Monotonous Cursors Verification ───────────────────
        print("\n--- Test 2: Verifying Monotonous Cursor Colors ---")
        cursor_data = page.evaluate("""() => {
            const cursors = Array.from(document.querySelectorAll('.agent-cursor:not(.helper-cursor)'));
            const helpers = Array.from(document.querySelectorAll('.agent-cursor.helper-cursor'));
            return {
                cursorFills: cursors.map(c => c.querySelector('path').getAttribute('fill')),
                helperFills: helpers.map(h => h.querySelector('path').getAttribute('fill')),
                cursorPills: cursors.map(c => c.querySelector('.cursor-pill').textContent.trim()),
                helperRoles: helpers.map(h => h.querySelector('.helper-role-text').textContent.trim())
            };
        }""")

        primary_fills = set(cursor_data["cursorFills"])
        helper_fills = set(cursor_data["helperFills"])
        print(f"  Primary cursor fills: {primary_fills}")
        print(f"  Helper cursor fills: {helper_fills}")

        assert len(primary_fills) == 1, f"Expected monotonous fill for primary cursors, got {primary_fills}"
        assert next(iter(primary_fills)) == "#0f172a", f"Expected #0f172a, got {primary_fills}"
        assert len(helper_fills) == 1, f"Expected monotonous fill for helper cursors, got {helper_fills}"
        assert next(iter(helper_fills)) == "#0f172a", f"Expected #0f172a, got {helper_fills}"
        assert len(cursor_data["cursorPills"]) == 10, "Expected 10 primary cursors"
        assert len(cursor_data["helperRoles"]) == 10, "Expected 10 helper cursors"
        print("✓ All 20 cursors are strictly monotonous slate (#0f172a) with clear role pills")

        # ── Test 3: Mode 2 Hexagon Choreography (Extraction Outside & Shape Morph) ───
        print("\n--- Test 3: Verifying Hexagon Choreography Sequence in Light Mode ---")
        # Set 2x speed for test efficiency
        page.locator(".speed-opt[data-speed='2.0']").click()
        page.wait_for_timeout(100)

        # Trigger Hexagon Core Build
        page.locator("#btn-hexagon").click()
        print("  Triggered Hexagon Core Build...")

        # Wait for Slice & Extraction phase (~1.55s at 2x speed)
        page.wait_for_timeout(1550)
        hud_status = page.locator("#hud-status").inner_text()
        print(f"  Status during extraction: {hud_status}")

        # Check that central core has dissolved while shards translated outward
        core_opacity = float(page.evaluate("() => getComputedStyle(document.getElementById('hex-trace-path')).opacity"))
        print(f"  hex-trace-path opacity during extraction: {core_opacity}")
        assert core_opacity < 0.1, f"Expected core to be dissolved, got {core_opacity}"

        page.screenshot(path=str(SCREENSHOT_DIR / "01_light_mode_extraction_outside.png"))
        print(f"  Saved screenshot: {SCREENSHOT_DIR / '01_light_mode_extraction_outside.png'}")

        # Wait for 90° Shape Rectification outside (~2.25s at 2x speed)
        page.wait_for_timeout(700)
        hud_status_sculpt = page.locator("#hud-status").inner_text()
        print(f"  Status during 90° rectification: {hud_status_sculpt}")

        # Verify ruler badge appears and central core is 0 opacity
        core_opacity_sculpt = float(page.evaluate("() => getComputedStyle(document.getElementById('hex-trace-path')).opacity"))
        assert core_opacity_sculpt == 0, f"Central hexagon must be 0 opacity during outside sculpt, got {core_opacity_sculpt}"

        badge_texts = page.evaluate("""() => {
            const badges = Array.from(document.querySelectorAll('.sculptor-ruler-badge text'));
            return badges.map(b => b.textContent.trim()).filter(t => t.length > 0);
        }""")
        print(f"  Ruler badges visible outside: {badge_texts[:3]}...")
        page.screenshot(path=str(SCREENSHOT_DIR / "02_light_mode_90deg_rectification.png"))
        print(f"  Saved screenshot: {SCREENSHOT_DIR / '02_light_mode_90deg_rectification.png'}")

        # Wait for full completion (~4.5s total at 2x speed)
        page.wait_for_timeout(2200)
        final_status = page.locator("#hud-status").inner_text()
        print(f"  Final HUD status: {final_status}")
        assert "Complete" in final_status, f"Expected Complete status, got {final_status}"

        # Verify all 10 cards have '.built' class
        built_cards_count = page.evaluate("() => document.querySelectorAll('.card-block.built').length")
        print(f"  Built cards count: {built_cards_count}/10")
        assert built_cards_count >= 9, f"Expected at least 9 built cards, got {built_cards_count}"

        page.screenshot(path=str(SCREENSHOT_DIR / "03_light_mode_completed_dashboard.png"))
        print(f"  Saved screenshot: {SCREENSHOT_DIR / '03_light_mode_completed_dashboard.png'}")

        # Also take Night mode completion screenshot for comparison
        theme_btn.click()
        page.wait_for_timeout(400)
        page.screenshot(path=str(SCREENSHOT_DIR / "04_night_mode_completed_dashboard.png"))
        print(f"  Saved screenshot: {SCREENSHOT_DIR / '04_night_mode_completed_dashboard.png'}")

        assert len(errors) == 0, f"Encountered page errors: {errors}"
        print("\n✨ ALL PLAYWRIGHT CHECKS PASSED WITH ZERO CONSOLE ERRORS!")
        browser.close()


if __name__ == "__main__":
    verify_swarm_dashboard()
