# Changelog

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/changelog.md  

All notable public API, protocol, and compatibility changes to Sorrel are documented in this file.

---

## [1.0.0] — 2026-10-12

### Added
- **External AI Discovery Surface**:
  - Root `/llms.txt` index adhering to standard machine-readable agent specification.
  - Public `sitemap.xml` enumerating canonical documentation endpoints.
  - Crawler policy `robots.txt` explicitly governing search indexers vs. AI training scrapers.
  - RFC 9116 security declaration at `/.well-known/security.txt` and `/security.txt`.
- **Public AI Documentation**:
  - Released comprehensive Markdown documentation across `docs/public/` covering product overview, seven-stage workflow, safety boundaries, API quickstart, MCP server, file limits, result schemas, and quotas.
- **Origin Configuration Engine**:
  - Introduced `src/core/origin_config.py` with environment variable override (`SORREL_PUBLIC_ORIGIN`) and non-local validation.
- **Surface Validation Script**:
  - Added `scripts/check_ai_surface.py` to continuously validate root assets, anchor integrity, and schema compliance in CI.
- **Discovery Test Suite**:
  - Implemented `tests/test_ai_discovery.py` verifying contract integrity and crawler compliance.
