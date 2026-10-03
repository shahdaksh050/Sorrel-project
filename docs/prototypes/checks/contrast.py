import os
import re

P = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "docs", "prototypes", "verdacert_dsa_preview.html")
s = open(P, encoding="utf-8").read()


def block(selector):
    a = s.index(selector)
    b = s.index("}", a)
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", s[a:b]))


day = block(":root {\n      /* Day Theme")
night = {**day, **block('[data-theme="night"] {\n      /* Night Theme')}


def rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def lin(c):
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lum(c):
    r, g, b = (lin(x) for x in rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def mix(fg, bg, pct):
    f, b = rgb(fg), rgb(bg)
    return "#%02x%02x%02x" % tuple(round(pct * x + (1 - pct) * y) for x, y in zip(f, b))


pairs = [
    (".fs.ok / .avail (positive on paper)", "--positive", "--paper", None),
    (".fs.risk / .cc .mk / .check.risk (danger-text on paper)", "--danger-text", "--paper", None),
    (".fs.note / .chart-cap / .avail.off (ink-3 on paper)", "--ink-3", "--paper", None),
    ("ink-3 on page bg (.ws-caption, .hint)", "--ink-3", "--bg", None),
    ("ink-3 on bg-alt (.avail.off, .artifact.off)", "--ink-3", "--bg-alt", None),
    ("ink-2 on paper (body in cards)", "--ink-2", "--paper", None),
    ("warning on paper (.gauge.flag .s, .ws-alert mark)", "--warning", "--paper", None),
        ("ink on 9% warning tint (.run-banner.caution)", "--ink", "--paper", ("--warning", 0.09)),
    ("ink on 10% warning tint (.ws-alert)", "--ink", "--paper", ("--warning", 0.10)),
    ("accent-text on accent-soft (.file-identity, .run-banner, .cert-pill)", "--accent-text", "--accent-soft", None),
    ("accent-ink on accent (.step-n)", "--accent-ink", "--accent", None),
    ("ink on accent-soft (.sc.active .nm)", "--ink", "--accent-soft", None),
    ("accent-text on page bg (.serif-it, links)", "--accent-text", "--bg", None),
    ("accent-text on paper (.cert-block-title)", "--accent-text", "--paper", None),
    ("danger-text on page bg", "--danger-text", "--bg", None),
    ("danger-text on accent-soft", "--danger-text", "--accent-soft", None),
]
worst = []
for theme, tokens in (("day", day), ("night", night)):
    print(f"--- {theme}")
    for label, fg, bg, tint in pairs:
        bgc = tokens[bg]
        if tint:
            bgc = mix(tokens[tint[0]], tokens[bg], tint[1])
        r = ratio(tokens[fg], bgc)
        flag = "" if r >= 4.5 else "   <-- UNDER 4.5"
        print(f"{r:5.2f}  {label}{flag}")
        if r < 4.5:
            worst.append((theme, label, round(r, 2)))
print("\nunder 4.5:1:", worst or "none")
