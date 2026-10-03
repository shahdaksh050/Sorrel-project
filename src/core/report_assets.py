"""
Self-contained assets for the shared HTML report, so it opens with no network.

The report used to load its fonts from Google and Vega from a CDN, so on a
disconnected machine charts were blank and the type fell back to the system
font. This module reads the vendored copies in `static/` and returns them as
inline markup:

* fonts: the Latin-subset `@font-face` rules of the Sorrel families (Geist, Geist
  Mono, Newsreader) from `static/fonts/ledger-fonts.css`, each `.woff2` embedded
  as a data URI. They are variable fonts: one file carries a whole weight range
  (`font-weight: 100 900`), which the embedded rule keeps. Faces that point at
  identical files (a static family declared once per weight) are embedded once,
  with the range of the weights they were declared at. Other families in that
  stylesheet are skipped without reading their files;
* Vega, Vega-Lite and vega-embed: the minified bundles from `static/vendor/vega/`,
  inline, only when the report has charts.

Each asset falls back on its own to the CDN tag the report used before, if its
files are missing. Reading is cached per root, so each file is read once.
Pure stdlib, no I/O beyond reading those files.
"""
from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Repo-level `static/` folder (module attribute so tests can point it elsewhere).
ASSET_ROOT: Path = Path(__file__).resolve().parents[2] / "static"

#: Fallback only, used when the font files are missing: the same three families from Google Fonts.
FONTS_CDN = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700'
    "&family=Geist+Mono:wght@400;500;600"
    "&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500"
    '&display=swap" rel="stylesheet">'
)

VEGA_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
)

#: Load order matters: vega-embed needs the other two on the page first.
_VEGA_FILES = ("vega.min.js", "vega-lite.min.js", "vega-embed.min.js")
_FONT_CSS = "ledger-fonts.css"
#: The only families the report embeds. Compared exactly (after unquoting), so "Geist" never matches "Geist Mono".
_REPORT_FAMILIES = frozenset({"Geist", "Geist Mono", "Newsreader"})
#: The Latin subset of a face: `name-latin.woff2` (static) or `name-latin-wght-normal|italic.woff2` (variable).
_LATIN_FILE_RE = re.compile(r"-latin(?:-wght-(?:normal|italic))?\.woff2$")
_FACE_RE = re.compile(r"@font-face\s*\{(?P<body>[^}]*)\}", re.DOTALL)


@dataclass(frozen=True)
class ReportAssets:
    """The head markup for one report, and whether it needs the network."""

    fonts_html: str
    vega_html: str
    #: True when every asset the report needs is inline.
    offline: bool


def _prop(body: str, name: str) -> str | None:
    match = re.search(rf"(?<![\w-]){name}\s*:\s*([^;]+);", body)
    return match.group(1).strip() if match else None


def _weights(value: str | None) -> list[int]:
    """The weights a `font-weight` value declares: `500` is one, `100 900` is a variable-font range."""
    return [int(token) for token in (value or "").split() if token.isdigit()]


@lru_cache(maxsize=8)
def _inline_fonts(root: Path) -> str | None:
    """`<style>` with the Latin `@font-face` rules embedded, or None if unavailable."""
    font_dir = root / "fonts"
    try:
        css = (font_dir / _FONT_CSS).read_text(encoding="utf-8")
    except OSError:
        return None

    # (family, style, sha) -> {"weights": [...], "range": str, "data": bytes}
    faces: dict[tuple[str, str, str], dict[str, object]] = {}
    for match in _FACE_RE.finditer(css):
        body = match.group("body")
        src = re.search(r"url\(['\"]?([^'\")]+)['\"]?\)", body)
        family = (_prop(body, "font-family") or "").strip("'\"")
        if src is None or family not in _REPORT_FAMILIES or not _LATIN_FILE_RE.search(src.group(1)):
            continue  # another family or a non-Latin subset: never read, so it cannot trigger the fallback
        try:
            data = (font_dir / src.group(1)).read_bytes()
        except OSError:
            return None  # a named file is missing: use the CDN rather than half the fonts
        style = _prop(body, "font-style") or "normal"
        key = (family, style, hashlib.sha256(data).hexdigest())
        face = faces.setdefault(
            key, {"weights": [], "range": _prop(body, "unicode-range") or "", "data": data}
        )
        weights = face["weights"]
        assert isinstance(weights, list)
        weights.extend(_weights(_prop(body, "font-weight")))
    if not faces:
        return None

    rules: list[str] = []
    for (family, style, _), face in faces.items():
        weights = face["weights"]
        assert isinstance(weights, list) and isinstance(face["data"], bytes)
        low, high = (min(weights), max(weights)) if weights else (400, 400)
        weight_css = str(low) if low == high else f"{low} {high}"
        uri = "data:font/woff2;base64," + base64.b64encode(face["data"]).decode("ascii")
        rule = (
            f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight_css};"
            f"font-display:swap;src:url({uri}) format('woff2');"
        )
        if face["range"]:
            rule += f"unicode-range:{face['range']};"
        rules.append(rule + "}")
    return "<style>" + "".join(rules) + "</style>"


@lru_cache(maxsize=8)
def _inline_vega(root: Path) -> str | None:
    """Three inline `<script>` tags with the vendored Vega bundles, or None if unavailable."""
    scripts: list[str] = []
    for name in _VEGA_FILES:
        try:
            code = (root / "vendor" / "vega" / name).read_text(encoding="utf-8")
        except OSError:
            return None
        # A literal "</script" would end the tag early; "<\/script" is the same string in JS.
        scripts.append("<script>" + re.sub(r"(?i)</(script)", r"<\\/\1", code) + "</script>")
    return "".join(scripts)


def report_assets(*, need_vega: bool, root: Path | None = None) -> ReportAssets:
    """Head assets for a report; `need_vega` is False for a report with no charts."""
    base = ASSET_ROOT if root is None else root
    fonts = _inline_fonts(base)
    fonts_html = fonts if fonts is not None else FONTS_CDN
    if not need_vega:
        return ReportAssets(fonts_html, "", offline=fonts is not None)
    vega = _inline_vega(base)
    return ReportAssets(
        fonts_html,
        vega if vega is not None else VEGA_CDN,
        offline=fonts is not None and vega is not None,
    )
