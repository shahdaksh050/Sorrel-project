# REST API Roadmap

Version: unreleased
Last Updated: 2026-10-16
Canonical: /docs/api-quickstart.md

## Availability

Sorrel does **not** currently ship a REST API. This repository provides a local
Streamlit application (`streamlit run app.py`) and a local command-line entry
point (`python main.py --dataset <path>`). There are no `/v1/` routes, bearer
tokens, asynchronous run handles, SSE events, artifact-download endpoints, or
HTTP quota responses in this release.

The endpoint sketches previously published here are roadmap material, not an
integration contract. Do not build against them.

## Future direction

Any remote API will be documented only after its implementation, authentication,
tenant isolation, retention behaviour, and schema conformance tests ship in the
same release.
