import pytest

from src.core.design_tokens import palette


def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c + c for c in hex_str)
    if len(hex_str) == 8:
        # ignore alpha for now, assuming 1.0 or mixing on opaque bg manually
        hex_str = hex_str[:6]
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)

def luminance(r8: int, g8: int, b8: int) -> float:
    def srgb_to_lin(c: int) -> float:
        c_norm = c / 255.0
        if c_norm <= 0.03928:
            return c_norm / 12.92
        else:
            return ((c_norm + 0.055) / 1.055) ** 2.4
    return 0.2126 * srgb_to_lin(r8) + 0.7152 * srgb_to_lin(g8) + 0.0722 * srgb_to_lin(b8)

def contrast(c1: str, c2: str) -> float:
    l1 = luminance(*hex_to_rgb(c1))
    l2 = luminance(*hex_to_rgb(c2))
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

def mix(color: str, amount: float, bg: str) -> str:
    c_rgb = hex_to_rgb(color)
    bg_rgb = hex_to_rgb(bg)
    mixed = (
        int(c_rgb[0] * amount + bg_rgb[0] * (1 - amount)),
        int(c_rgb[1] * amount + bg_rgb[1] * (1 - amount)),
        int(c_rgb[2] * amount + bg_rgb[2] * (1 - amount))
    )
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"

def test_contrast_ratios():
    print("\\n=== WCAG Contrast Ratios ===")
    print(f"{'Pair':<30} | {'Day':<8} | {'Night':<8}")
    print("-" * 52)

    day = palette("day")
    night = palette("night")

    # Add derived tokens
    # ui/styles.py definitions:
    # --rule-strong: color-mix(in srgb, var(--rule) 60%, #000) for day, 50% #fff for night
    day["rule_strong"] = mix(day["rule"], 0.60, "#000000")
    night["rule_strong"] = mix(night["rule"], 0.50, "#ffffff")

    # --risk-text: var(--risk) for day, #ff8a8a for night
    day["risk_text"] = day["risk"]
    night["risk_text"] = "#ff8a8a"

    # --glow (pen @ 12% on sheet for day, 15% on sheet for night)
    day["glow_on_sheet"] = mix(day["pen"], 0.12, day["sheet"])
    night["glow_on_sheet"] = mix(night["pen"], 0.15, night["sheet"])

    # --spot (accent @ 12% on sheet for day, 15% on sheet for night)
    day["spot_on_sheet"] = mix(day["accent"], 0.12, day["sheet"])
    night["spot_on_sheet"] = mix(night["accent"], 0.15, night["sheet"])

    tests = [
        # Normal text (AA requires 4.5:1)
        ("ink on stock", "ink", "stock", 4.5),
        ("ink on sheet", "ink", "sheet", 4.5),
        ("ink on sheet_alt", "ink", "sheet_alt", 4.5),
        ("graphite on sheet", "graphite", "sheet", 4.5),
        ("pen on sheet", "pen", "sheet", 4.5),
        ("risk_text on sheet", "risk_text", "sheet", 4.5),

        # UI components (AA requires 3:1)
        ("rule_strong on stock", "rule_strong", "stock", 3.0),
        ("rule_strong on sheet", "rule_strong", "sheet", 3.0),

        # Texts on tints
        ("risk_text on risk tint", "risk_text", "glow_on_sheet", 4.5),

        # Verify accent is decorative only (fails AA for text)
        ("accent on sheet", "accent", "sheet", -1), # -1 means just report, don't assert pass
    ]

    failures = []
    for name, fg, bg, req in tests:
        c_day = contrast(day[fg], day[bg])
        c_night = contrast(night[fg], night[bg])
        print(f"{name:<30} | {c_day:5.2f}:1 | {c_night:5.2f}:1")
        if req > 0:
            if c_day < req:
                failures.append(f"Day {name} ({c_day:.2f}) < {req}")
            if c_night < req:
                failures.append(f"Night {name} ({c_night:.2f}) < {req}")

    if failures:
        pytest.fail("\\n".join(failures))
