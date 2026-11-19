"""Public discovery describes only the shipped local Streamlit/CLI product."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.check_ai_surface import embedded_docs_registry, main

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = REPO_ROOT / "public"
DOCS_PUBLIC_DIR = REPO_ROOT / "docs" / "public"
ROADMAP_DOCS = ("api-quickstart.md", "mcp.md", "result-schema.md")


def test_surface_linter_script() -> None:
    assert main([]) == 0


def test_embedded_docs_registry_matches_public_sources() -> None:
    registry = embedded_docs_registry()
    assert registry["overview"] == (DOCS_PUBLIC_DIR / "overview.md").read_text(encoding="utf-8")
    assert "localhost" not in "\n".join(registry.values()).lower()


def test_landing_embeds_truthful_deployment_safe_public_text() -> None:
    landing = (REPO_ROOT / "ui" / "landing_component" / "index.html").read_text(encoding="utf-8")
    assert "localhost" not in landing.lower()
    assert "REST API is not shipped" in landing
    assert "Remote MCP is not shipped" in landing


def test_discovery_is_deployment_safe_and_local_only() -> None:
    searchable = (PUBLIC_DIR / "llms.txt").read_text(encoding="utf-8")
    assert "There is no REST API or MCP server" in searchable
    assert "localhost" not in searchable.lower()
    assert "file:///" not in searchable.lower()
    for name in ROADMAP_DOCS:
        assert f"/docs/{name}" not in searchable

    for path in [*PUBLIC_DIR.glob("*.txt"), PUBLIC_DIR / ".well-known" / "security.txt"]:
        content = path.read_text(encoding="utf-8").lower()
        assert "localhost" not in content
        assert "file:///" not in content


def test_sitemap_uses_deployment_token_and_omits_unshipped_interfaces() -> None:
    root = ET.fromstring((PUBLIC_DIR / "sitemap.xml").read_text(encoding="utf-8"))
    locations = ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
    urls = [item.text or "" for item in root.findall(locations)]
    assert urls
    assert all(url.startswith("{{PUBLIC_ORIGIN}}/") for url in urls)
    assert all(name not in "\n".join(urls) for name in ROADMAP_DOCS)


def test_public_docs_use_root_relative_canonicals_and_truthful_boundaries() -> None:
    for path in DOCS_PUBLIC_DIR.glob("*.md"):
        content = path.read_text(encoding="utf-8")
        assert f"Canonical: /docs/{path.name}" in content
        assert "localhost" not in content.lower()
        assert "file:///" not in content.lower()

    workflow = (DOCS_PUBLIC_DIR / "workflow.md").read_text(encoding="utf-8").lower()
    assert "five stages" in workflow
    assert "local analysis worker" in workflow
    safety = (DOCS_PUBLIC_DIR / "safety.md").read_text(encoding="utf-8").lower()
    assert "mutable local files" in safety
    for name in ROADMAP_DOCS:
        content = (DOCS_PUBLIC_DIR / name).read_text(encoding="utf-8").lower()
        assert "not" in content
