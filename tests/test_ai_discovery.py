"""
Tests for AI Discovery Surface and Public Compatibility Contract

Validates robots.txt, sitemap.xml, llms.txt, security.txt, public Markdown
documentation, canonical origin configuration, and indexable landing metadata.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.core.origin_config import (
    DEFAULT_ORIGIN,
    ENV_VAR_ORIGIN,
    build_alternate_link_tag,
    build_canonical_url,
    build_describedby_link_tag,
    build_doc_url,
    get_public_origin,
    is_local_origin,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = REPO_ROOT / "public"
DOCS_PUBLIC_DIR = REPO_ROOT / "docs" / "public"
LANDING_INDEX = REPO_ROOT / "ui" / "landing_component" / "index.html"


class TestOriginConfig:
    def test_default_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR_ORIGIN, raising=False)
        assert get_public_origin() == DEFAULT_ORIGIN
        assert get_public_origin() == "http://localhost:8501"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR_ORIGIN, "https://analytics.mycorp.com/")
        assert get_public_origin() == "https://analytics.mycorp.com"

    def test_protocol_normalization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR_ORIGIN, "sorrel.example.com")
        assert get_public_origin() == "https://sorrel.example.com"

    def test_build_canonical_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR_ORIGIN, raising=False)
        assert build_canonical_url() == "http://localhost:8501/"
        assert build_canonical_url("docs/overview.md") == "http://localhost:8501/docs/overview.md"
        assert build_canonical_url("/api/v1/runs") == "http://localhost:8501/api/v1/runs"

    def test_build_doc_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR_ORIGIN, raising=False)
        assert build_doc_url("overview") == "http://localhost:8501/docs/overview.md"
        assert build_doc_url("workflow.md") == "http://localhost:8501/docs/workflow.md"

    def test_link_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_VAR_ORIGIN, raising=False)
        alt = build_alternate_link_tag("overview")
        assert 'rel="alternate"' in alt
        assert 'type="text/markdown"' in alt
        assert 'href="http://localhost:8501/docs/overview.md"' in alt

        desc = build_describedby_link_tag()
        assert 'rel="describedby"' in desc
        assert 'href="http://localhost:8501/llms.txt"' in desc

    def test_is_local_origin(self) -> None:
        assert is_local_origin("http://localhost:8501") is True
        assert is_local_origin("http://127.0.0.1:8000") is True
        assert is_local_origin("https://dsa-agent.local") is True
        assert is_local_origin("https://sorrel.ai") is False


class TestPublicRobotsTxt:
    def test_robots_txt_exists_and_valid(self) -> None:
        robots_file = PUBLIC_DIR / "robots.txt"
        assert robots_file.is_file()
        content = robots_file.read_text(encoding="utf-8")

        assert "User-agent: *" in content
        assert "Disallow: /api/" in content
        assert "Disallow: /runs/" in content
        assert "Disallow: /uploads/" in content
        assert "Sitemap: http://localhost:8501/sitemap.xml" in content

    def test_crawler_matrix_policy(self) -> None:
        content = (PUBLIC_DIR / "robots.txt").read_text(encoding="utf-8")
        # Search bot allowed
        assert "User-agent: OAI-SearchBot\nAllow: /" in content
        # Training bots declined
        assert "User-agent: GPTBot\nDisallow: /" in content
        assert "User-agent: Google-Extended\nDisallow: /" in content
        assert "User-agent: CCBot\nDisallow: /" in content

        # No local artifacts
        assert ".local" not in content


class TestPublicSitemapXml:
    def test_sitemap_xml_parses_cleanly(self) -> None:
        sitemap_file = PUBLIC_DIR / "sitemap.xml"
        assert sitemap_file.is_file()
        content = sitemap_file.read_text(encoding="utf-8")

        root = ET.fromstring(content)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [elem.text.strip() for elem in root.findall(".//sm:loc", namespace) if elem.text]

        assert "http://localhost:8501/" in urls
        assert "http://localhost:8501/docs/overview.md" in urls
        assert "http://localhost:8501/docs/workflow.md" in urls
        assert "http://localhost:8501/docs/safety.md" in urls
        assert "http://localhost:8501/docs/api-quickstart.md" in urls
        assert "http://localhost:8501/docs/mcp.md" in urls
        assert "http://localhost:8501/docs/file-formats.md" in urls
        assert "http://localhost:8501/docs/result-schema.md" in urls
        assert "http://localhost:8501/docs/limits.md" in urls
        assert "http://localhost:8501/docs/changelog.md" in urls

        # Zero private URLs
        for url in urls:
            assert "/api/" not in url
            assert "/runs/" not in url
            assert "/uploads/" not in url
            assert ".local" not in url


class TestPublicLlmsTxt:
    def test_llms_txt_structure_and_headings(self) -> None:
        llms_file = PUBLIC_DIR / "llms.txt"
        assert llms_file.is_file()
        content = llms_file.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Rule 1: Exactly 1 H1 heading
        h1s = [line for line in lines if line.startswith("# ") and not line.startswith("## ")]
        assert len(h1s) == 1
        assert h1s[0] == "# Sorrel"

        # Rule 2: Blockquote summary
        assert any(line.startswith("> Sorrel is an evidence-first data-analysis service.") for line in lines)

        # Rule 3: Required sections
        assert "## Start Here" in content
        assert "## Integrate" in content
        assert "## Reference" in content
        assert "## Optional" in content

        # Rule 4: Compact budget
        assert len(content.encode("utf-8")) < 4096

    def test_llms_txt_all_links_resolve(self) -> None:
        content = (PUBLIC_DIR / "llms.txt").read_text(encoding="utf-8")
        link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
        matches = list(link_pattern.finditer(content))
        assert len(matches) >= 8

        for match in matches:
            url = match.group(2)
            assert url.startswith("http://localhost:8501/")
            if url.endswith(".md"):
                slug = url.split("/docs/")[-1]
                target_file = DOCS_PUBLIC_DIR / slug
                assert target_file.is_file(), f"Linked documentation file {target_file} does not exist"


class TestSecurityTxt:
    def test_security_txt_conformance(self) -> None:
        for path in [PUBLIC_DIR / ".well-known" / "security.txt", PUBLIC_DIR / "security.txt"]:
            assert path.is_file()
            content = path.read_text(encoding="utf-8")
            assert "Contact: mailto:security@sorrel.ai" in content
            assert "Canonical: http://localhost:8501/.well-known/security.txt" in content
            assert "Policy: http://localhost:8501/docs/safety.md" in content
            assert "Preferred-Languages: en" in content


class TestPublicDocumentationPages:
    def test_required_docs_exist_and_conform(self) -> None:
        required_pages = [
            "overview.md",
            "workflow.md",
            "safety.md",
            "api-quickstart.md",
            "mcp.md",
            "file-formats.md",
            "result-schema.md",
            "limits.md",
            "changelog.md",
        ]
        for name in required_pages:
            doc_path = DOCS_PUBLIC_DIR / name
            assert doc_path.is_file(), f"Missing public doc {name}"
            text = doc_path.read_text(encoding="utf-8")
            assert text.startswith("# "), f"{name} must begin with an H1 heading"
            assert "Version:" in text, f"{name} must contain Version header"
            assert "Last Updated:" in text, f"{name} must contain Last Updated header"
            assert "Canonical: http://localhost:8501/docs/" in text, f"{name} must contain Canonical header"
            assert ".local" not in text


class TestLandingIndexMetadata:
    def test_landing_index_metadata(self) -> None:
        content = LANDING_INDEX.read_text(encoding="utf-8")
        assert "dsa-agent.local" not in content
        assert '<link rel="canonical" href="http://localhost:8501/">' in content
        assert '<meta property="og:url" content="http://localhost:8501/">' in content
        assert '<link rel="alternate" type="text/markdown" href="http://localhost:8501/docs/overview.md">' in content
        assert '<link rel="describedby" href="http://localhost:8501/llms.txt">' in content


def test_surface_linter_script() -> None:
    from scripts.check_ai_surface import main

    assert main([]) == 0
