"""
Origin Configuration & Canonical URL Management

Centralizes public deployment origin management for Sorrel, ensuring
metadata, sitemaps, robots.txt, and HTML link tags always resolve to
valid, non-local canonical endpoints.
"""
from __future__ import annotations

import os
import re
from typing import Final

DEFAULT_ORIGIN: Final[str] = "http://localhost:8501"
ENV_VAR_ORIGIN: Final[str] = "SORREL_PUBLIC_ORIGIN"

# Pattern detecting placeholder or local development origins
_LOCAL_ORIGIN_PATTERN = re.compile(r"^(https?://)?(localhost|127\.0\.0\.1|.*\.local)(:\d+)?$", re.IGNORECASE)


def get_public_origin() -> str:
    """Return the configured public deployment origin for Sorrel.

    Reads from the ``SORREL_PUBLIC_ORIGIN`` environment variable, falling back
    to ``http://localhost:8501``. Trailing slashes are stripped.
    """
    raw = os.environ.get(ENV_VAR_ORIGIN, "").strip()
    if not raw:
        return DEFAULT_ORIGIN
    cleaned = raw.rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        scheme = "http" if ("localhost" in cleaned or "127.0.0.1" in cleaned) else "https"
        cleaned = f"{scheme}://{cleaned}"
    return cleaned


def is_local_origin(origin: str | None = None) -> bool:
    """Check whether the configured or supplied origin represents a local test host."""
    target = origin or get_public_origin()
    return bool(_LOCAL_ORIGIN_PATTERN.match(target))


def build_canonical_url(path: str = "/") -> str:
    """Construct an absolute canonical URL using the configured public origin.

    Args:
        path: Relative path beginning with or without a slash (e.g. ``/docs/overview.md``).

    Returns:
        Absolute URL (e.g. ``http://localhost:8501/docs/overview.md``).
    """
    origin = get_public_origin()
    clean_path = "/" + path.lstrip("/") if path else "/"
    return f"{origin}{clean_path}"


def build_doc_url(doc_slug: str) -> str:
    """Construct an absolute canonical URL for a public documentation page."""
    slug = doc_slug.strip().lstrip("/")
    if not slug.endswith(".md"):
        slug = f"{slug}.md"
    return build_canonical_url(f"/docs/{slug}")


def build_alternate_link_tag(doc_slug: str) -> str:
    """Generate an HTML <link rel="alternate"> tag pointing to the clean Markdown document."""
    url = build_doc_url(doc_slug)
    return f'<link rel="alternate" type="text/markdown" href="{url}">'


def build_describedby_link_tag() -> str:
    """Generate an HTML <link rel="describedby"> tag referencing the root /llms.txt."""
    url = build_canonical_url("/llms.txt")
    return f'<link rel="describedby" href="{url}">'


def sync_discovery_assets_to_static() -> int:
    """Synchronize public/ discovery files and docs into Streamlit's root static directory.

    Ensures direct HTTP requests to /robots.txt, /sitemap.xml, /llms.txt,
    /.well-known/security.txt, and /docs/*.md return actual machine-readable assets
    rather than falling back to Streamlit's index.html.
    """
    import shutil
    from pathlib import Path

    try:
        from streamlit import file_util

        static_dir = Path(file_util.get_static_dir())
    except Exception:
        return 0

    if not static_dir.is_dir():
        return 0

    repo_root = Path(__file__).resolve().parent.parent.parent
    public_dir = repo_root / "public"
    docs_dir = repo_root / "docs" / "public"
    synced = 0

    if public_dir.is_dir():
        for item in public_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(public_dir)
                dest = static_dir / rel
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists() or dest.stat().st_mtime < item.stat().st_mtime:
                        shutil.copy2(item, dest)
                        synced += 1
                except OSError:
                    pass

    if docs_dir.is_dir():
        for item in docs_dir.glob("*.md"):
            dest = static_dir / "docs" / item.name
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or dest.stat().st_mtime < item.stat().st_mtime:
                    shutil.copy2(item, dest)
                    synced += 1
            except OSError:
                pass

    return synced

