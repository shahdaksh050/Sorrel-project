#!/usr/bin/env python3
"""
AI Discovery Surface Checker & Linter

Validates public discovery files (robots.txt, sitemap.xml, llms.txt, security.txt)
and public Markdown documentation against repository standards, ensuring zero
hallucinated paths, invalid local URLs, or broken links.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = REPO_ROOT / "public"
DOCS_PUBLIC_DIR = REPO_ROOT / "docs" / "public"
LANDING_INDEX = REPO_ROOT / "ui" / "landing_component" / "index.html"


def check_robots_txt(origin: str) -> list[str]:
    errors: list[str] = []
    robots_path = PUBLIC_DIR / "robots.txt"
    if not robots_path.is_file():
        return [f"Missing {robots_path}"]

    content = robots_path.read_text(encoding="utf-8")
    if "User-agent:" not in content:
        errors.append("robots.txt must declare at least one User-agent directive")
    if "Disallow: /api/" not in content or "Disallow: /runs/" not in content:
        errors.append("robots.txt must disallow private paths (/api/, /runs/, /uploads/)")
    if "Sitemap:" not in content:
        errors.append("robots.txt must specify a Sitemap URL")
    if ".local" in content or ("localhost" in content and not origin.startswith("http://localhost")):
        errors.append("robots.txt contains illegal .local or localhost references")

    expected_sitemap = f"{origin}/sitemap.xml"
    if expected_sitemap not in content:
        errors.append(f"robots.txt Sitemap directive should match canonical origin ({expected_sitemap})")

    return errors


def check_sitemap_xml(origin: str) -> list[str]:
    errors: list[str] = []
    sitemap_path = PUBLIC_DIR / "sitemap.xml"
    if not sitemap_path.is_file():
        return [f"Missing {sitemap_path}"]

    content = sitemap_path.read_text(encoding="utf-8")
    if ".local" in content or ("localhost" in content and not origin.startswith("http://localhost")):
        errors.append("sitemap.xml contains illegal .local or localhost references")

    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        return [f"sitemap.xml failed XML parsing: {e}"]

    # Extract all <loc> text values
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [elem.text.strip() for elem in root.findall(".//sm:loc", namespace) if elem.text]

    if not urls:
        errors.append("sitemap.xml contains no <loc> URL elements")

    for url in urls:
        if not url.startswith(origin):
            errors.append(f"sitemap URL does not start with expected origin ({origin}): {url}")
        if url.endswith(".md"):
            # Ensure referenced doc exists in docs/public/
            slug = url.split("/docs/")[-1]
            doc_file = DOCS_PUBLIC_DIR / slug
            if not doc_file.is_file():
                errors.append(f"sitemap.xml references missing documentation file: {doc_file}")

    return errors


def check_llms_txt(origin: str) -> list[str]:
    errors: list[str] = []
    llms_path = PUBLIC_DIR / "llms.txt"
    if not llms_path.is_file():
        return [f"Missing {llms_path}"]

    content = llms_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Rule 1: Exactly one H1 heading
    h1_count = sum(1 for line in lines if line.startswith("# ") and not line.startswith("## "))
    if h1_count != 1:
        errors.append(f"llms.txt must have exactly 1 H1 heading (found {h1_count})")

    # Rule 2: Must contain blockquote summary
    if not any(line.startswith("> ") for line in lines):
        errors.append("llms.txt must have a blockquote summary (> ...)")

    # Rule 3: Required section headings
    required_sections = ["## Start Here", "## Integrate", "## Reference"]
    for sec in required_sections:
        if sec not in content:
            errors.append(f"llms.txt missing required section '{sec}'")

    # Rule 4: Context size budget (< 4096 bytes)
    if len(content.encode("utf-8")) > 4096:
        errors.append(f"llms.txt exceeds context size budget ({len(content.encode('utf-8'))} > 4096 bytes)")

    # Rule 5: Validate linked files
    link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    for match in link_pattern.finditer(content):
        title, url = match.group(1), match.group(2)
        if ".local" in url or ("localhost" in url and not origin.startswith("http://localhost")):
            errors.append(f"llms.txt link contains .local or localhost: {url}")
        if url.endswith(".md") and "/docs/" in url:
            slug = url.split("/docs/")[-1]
            doc_file = DOCS_PUBLIC_DIR / slug
            if not doc_file.is_file():
                errors.append(f"llms.txt links to missing documentation file '{title}': {doc_file}")

    return errors


def check_security_txt(origin: str) -> list[str]:
    errors: list[str] = []
    for sec_path in [PUBLIC_DIR / ".well-known" / "security.txt", PUBLIC_DIR / "security.txt"]:
        if not sec_path.is_file():
            errors.append(f"Missing {sec_path}")
            continue
        content = sec_path.read_text(encoding="utf-8")
        if "Contact: mailto:" not in content:
            errors.append(f"{sec_path.name} must declare a Contact directive")
        if "Canonical:" not in content:
            errors.append(f"{sec_path.name} must declare a Canonical directive")
        if "Expires:" not in content:
            errors.append(f"{sec_path.name} must declare an Expires directive")
        if ".local" in content or ("localhost" in content and not origin.startswith("http://localhost")):
            errors.append(f"{sec_path.name} contains .local or localhost")
    return errors


def check_public_docs(origin: str) -> list[str]:
    errors: list[str] = []
    if not DOCS_PUBLIC_DIR.is_dir():
        return [f"Missing {DOCS_PUBLIC_DIR} directory"]

    doc_files = list(DOCS_PUBLIC_DIR.glob("*.md"))
    if len(doc_files) < 6:
        errors.append(f"Expected at least 6 public documentation pages, found {len(doc_files)}")

    for doc in doc_files:
        text = doc.read_text(encoding="utf-8")
        if not text.startswith("# "):
            errors.append(f"{doc.name} should start with an H1 heading ('# Title')")
        if "Version:" not in text:
            errors.append(f"{doc.name} missing 'Version:' header")
        if "Last Updated:" not in text:
            errors.append(f"{doc.name} missing 'Last Updated:' header")
        if "Canonical:" not in text:
            errors.append(f"{doc.name} missing 'Canonical:' link header")
        if ".local" in text or ("localhost" in text and not origin.startswith("http://localhost")):
            errors.append(f"{doc.name} contains illegal .local or localhost reference")

    return errors


def check_landing_metadata() -> list[str]:
    errors: list[str] = []
    if not LANDING_INDEX.is_file():
        return [f"Missing {LANDING_INDEX}"]

    content = LANDING_INDEX.read_text(encoding="utf-8")
    if "dsa-agent.local" in content:
        errors.append("landing_component/index.html still references dsa-agent.local")
    if 'rel="canonical"' not in content:
        errors.append("landing_component/index.html missing rel=canonical tag")
    if 'rel="alternate"' not in content:
        errors.append("landing_component/index.html missing rel=alternate tag for Markdown docs")
    if 'rel="describedby"' not in content:
        errors.append("landing_component/index.html missing rel=describedby tag for llms.txt")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate public AI discovery surface.")
    parser.add_argument("--origin", default="http://localhost:8501", help="Expected canonical origin")
    args = parser.parse_args(argv)

    origin = args.origin.rstrip("/")
    all_errors: list[str] = []

    print(f"Checking AI discovery surface for origin: {origin}")

    all_errors.extend(check_robots_txt(origin))
    all_errors.extend(check_sitemap_xml(origin))
    all_errors.extend(check_llms_txt(origin))
    all_errors.extend(check_security_txt(origin))
    all_errors.extend(check_public_docs(origin))
    all_errors.extend(check_landing_metadata())

    if all_errors:
        print("\nDiscovery surface verification FAILED with the following errors:")
        for err in all_errors:
            print(f"  [ERROR] {err}")
        return 1

    print("\nAll discovery surface checks passed cleanly (0 errors)!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
