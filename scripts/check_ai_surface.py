#!/usr/bin/env python3
"""Validate the public discovery surface against shipped local capabilities."""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = REPO_ROOT / "public"
DOCS_PUBLIC_DIR = REPO_ROOT / "docs" / "public"
DISCOVERY_DOCS = {
    "overview.md", "workflow.md", "safety.md", "file-formats.md", "limits.md", "changelog.md",
}
ROADMAP_DOCS = {"api-quickstart.md", "mcp.md", "result-schema.md"}
FORBIDDEN_URL = re.compile(
    r"(?:https?://(?:localhost|127\\.0\\.0\\.1)|file:///|\\.local)",
    re.IGNORECASE,
)
REMOTE_CLAIMS = (
    "REST API enables", "provides a remote Model Context Protocol", "stable JSON schema returned",
    "active runs per tenant", "automatically purged after 24 hours", "Immutable Audit Trail",
)
LANDING_PATH = REPO_ROOT / "ui" / "landing_component" / "index.html"
_DOCS_REGISTRY_RE = re.compile(r"var DOCS_REGISTRY = .*?;\r?\n", re.DOTALL)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def embedded_docs_registry() -> dict[str, str]:
    """Return the deterministic documentation payload embedded by the UI."""
    return {
        path.stem: _read(path)
        for path in sorted(DOCS_PUBLIC_DIR.glob("*.md"), key=lambda item: item.name)
    }


def sync_landing_docs() -> None:
    """Replace the landing page's generated docs registry from source docs."""
    html = _read(LANDING_PATH)
    replacement = f"var DOCS_REGISTRY = {json.dumps(embedded_docs_registry(), ensure_ascii=False)};\n"
    updated, count = _DOCS_REGISTRY_RE.subn(lambda _: replacement, html, count=1)
    if count != 1:
        raise RuntimeError("landing page DOCS_REGISTRY marker was not found exactly once")
    blocks = {
        "llms-code-content": _read(PUBLIC_DIR / "llms.txt"),
        "robots-code-content": _read(PUBLIC_DIR / "robots.txt"),
        "sitemap-code-content": _read(PUBLIC_DIR / "sitemap.xml"),
        "policy-code-content": _read(PUBLIC_DIR / ".well-known" / "security.txt"),
        "api-code-content": (
            "REST API is not shipped in this repository.\n"
            "Use the local Streamlit workspace or: python main.py --dataset <path>\n"
            "See /docs/api-quickstart.md for the release-gated roadmap."
        ),
        "mcp-code-content": (
            "Remote MCP is not shipped in this repository.\n"
            "No /mcp endpoint or bearer-token contract is currently available.\n"
            "See /docs/mcp.md for the release-gated roadmap."
        ),
    }
    for element_id, content in blocks.items():
        pattern = re.compile(
            rf'(<pre class="ai-code-block" id="{re.escape(element_id)}">).*?(</pre>)',
            re.DOTALL,
        )
        updated, block_count = pattern.subn(
            lambda match, block=content: f"{match.group(1)}{escape(block)}{match.group(2)}",
            updated,
            count=1,
        )
        if block_count != 1:
            raise RuntimeError(f"landing page block {element_id!r} was not found exactly once")
    for old, new in (
        ('href="http://localhost:8501/"', 'href="/"'),
        ('href="http://localhost:8501/docs/overview.md"', 'href="/docs/overview.md"'),
        ('href="http://localhost:8501/llms.txt"', 'href="/llms.txt"'),
        ('content="http://localhost:8501/"', 'content="/"'),
    ):
        updated = updated.replace(old, new)
    LANDING_PATH.write_text(updated, encoding="utf-8")


def check_public_assets() -> list[str]:
    errors: list[str] = []
    assets = (
        PUBLIC_DIR / "robots.txt",
        PUBLIC_DIR / "llms.txt",
        PUBLIC_DIR / "security.txt",
        PUBLIC_DIR / ".well-known" / "security.txt",
    )
    for path in assets:
        if not path.is_file():
            errors.append(f"Missing {path.relative_to(REPO_ROOT)}")
            continue
        if FORBIDDEN_URL.search(_read(path)):
            errors.append(f"{path.relative_to(REPO_ROOT)} leaks a workstation URL")

    robots = _read(PUBLIC_DIR / "robots.txt")
    required_directives = (
        "User-agent: *",
        "Disallow: /api/",
        "Disallow: /runs/",
        "Sitemap: /sitemap.xml",
    )
    for directive in required_directives:
        if directive not in robots:
            errors.append(f"robots.txt missing {directive!r}")

    for path in (PUBLIC_DIR / "security.txt", PUBLIC_DIR / ".well-known" / "security.txt"):
        content = _read(path)
        required_directives = (
            "Contact: mailto:",
            "Canonical: /.well-known/security.txt",
            "Policy: /docs/safety.md",
            "Expires:",
        )
        for directive in required_directives:
            if directive not in content:
                errors.append(f"{path.relative_to(REPO_ROOT)} missing {directive!r}")

    sitemap = PUBLIC_DIR / "sitemap.xml"
    try:
        root = ET.fromstring(_read(sitemap))
    except ET.ParseError as exc:
        errors.append(f"sitemap.xml failed XML parsing: {exc}")
    else:
        locations = ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
        urls = [item.text or "" for item in root.findall(locations)]
        if not urls or any(not url.startswith("{{PUBLIC_ORIGIN}}/") for url in urls):
            errors.append("sitemap.xml must use the deployment-origin token")
        if any(name in "\n".join(urls) for name in ROADMAP_DOCS):
            errors.append("sitemap.xml must not index unshipped interface roadmap pages")
    return errors


def check_docs() -> list[str]:
    errors: list[str] = []
    for path in DOCS_PUBLIC_DIR.glob("*.md"):
        text = _read(path)
        if not text.startswith("# "):
            errors.append(f"{path.name} must begin with an H1")
        for header in ("Version:", "Last Updated:", f"Canonical: /docs/{path.name}"):
            if header not in text:
                errors.append(f"{path.name} missing {header!r}")
        if FORBIDDEN_URL.search(text):
            errors.append(f"{path.name} leaks a workstation URL")
        if path.name in DISCOVERY_DOCS and any(claim in text for claim in REMOTE_CLAIMS):
            errors.append(f"{path.name} claims an unshipped remote capability")
        if path.name in ROADMAP_DOCS and "not" not in text.lower():
            errors.append(f"{path.name} must clearly state that the interface is unshipped")
    llms = _read(PUBLIC_DIR / "llms.txt")
    if "There is no REST API or MCP server" not in llms:
        errors.append("llms.txt must state the local-only interface boundary")
    if any(f"/docs/{name}" in llms for name in ROADMAP_DOCS):
        errors.append("llms.txt must not link unshipped interface roadmap pages")
    landing = _read(LANDING_PATH)
    if FORBIDDEN_URL.search(landing):
        errors.append("landing page leaks a workstation URL")
    if "REST API is not shipped" not in landing or "Remote MCP is not shipped" not in landing:
        errors.append("landing page must label remote interfaces as unshipped")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--sync-landing" in args:
        sync_landing_docs()
    errors = check_public_assets() + check_docs()
    if errors:
        print("AI discovery surface verification FAILED:")
        for error in errors:
            print(f"  [ERROR] {error}")
        return 1
    print("AI discovery surface verification passed (0 errors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
