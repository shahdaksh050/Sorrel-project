# DSA Agent frontend: implementation plan

Status: ready to execute from batch 0. Nothing in this file is implemented yet
except what section 1 lists as done.

Date: 2026-10-02. Branch: `frontend-overhaul`.

This file is the step-by-step "how". `FrontendOverhaulPlan.md` stays the "why"
and the rule book: scope tiers (its section 2d), decisions (2b), design rules
(3 and 11), risks (15). Cite it by section number instead of copying it. Where
the two files disagree, the decisions log in section 2b of the overhaul plan
wins, and this file is edited.

Decisions that shape every batch below (overhaul plan section 2b):

- final-year project, laptop demo with a hosted link as backup;
- no visual direction board; the current buff-green palette stays;
- landing page gets copy fixes only; a rebuild is "Later";
- the plate and the cinematic export stay as they are for the viva;
- the next batch is the results object, "How we got here", answers-first
  ordering, the theme-file fix and removal of `ui/animations.py`.

## 0. Rules for whoever implements this

- Work on `frontend-overhaul`. One concern per commit, and ask the owner before
  each commit. No co-author trailer. `README.md` is the owner's; never stage it.
- Tag the branch before each batch (`git tag batch-N-start`). An uncommitted
  round of UI work was lost to a reset once; tags and commits are the protection.
- Gates for every commit that touches Python: `ruff check .` and `mypy src/`.
  Every new module under `src/` gets a test file in `tests/` with at least one
  planted case and one absent case.
- Run the app or the tests only when the owner says so. Static gates are always
  allowed.
- Ask-First items in `AGENTS.md`: `src/rlm/engine.py`, task decomposition, the
  `MemorySystem` schema, deleting source files, adding a tool. Batch 3 touches
  `controller.py` but not decomposition. Deleting `ui/animations.py` and
  `frontend-landing/` is a deletion of verified-dead or duplicate files and is
  covered by the owner's decisions; still search for importers first.
- Known and accepted deviation: with the UI-selected provider no longer in the
  environment, `llm_concurrency()` in `src/rlm/engine.py` uses the cloud default
  of 4 parallel decomposition calls even for a local model (decomposition is off
  by default). The engine was left alone on purpose.
- Style: sentence case, no em-dashes, no emoji in UI text or in this file's
  successors. Every `st.markdown(..., unsafe_allow_html=True)` string escapes
  dynamic text with `html.escape`.
- Reference by function name and a grep anchor, never by line number. Line
  numbers in `app.py` moved twice in one day.

## 1. Where things stand (verified 2026-10-02)

Done and owner-confirmed on `frontend-overhaul` (commits `77424d9`, `ffc4280`):

- `RunConfig` and a per-thread objective scope; the UI no longer writes provider,
  model, key, URL, objective, output directory or switches to `os.environ`.
- `DSA_HOSTED` flag: hides the local provider, ignores the server key, shows
  privacy and code-execution limits read-only.
- A real no-AI sample run (`_sample_run` flag) replaces the fabricated demo; the
  run gate is `has_key or not use_llm`; run directories are prefixed `dsa-run-`
  and removed on reset; run comparison is local-only.
- The owner committed a new `README.md` (`a9fd98e`).

Verified harness to reuse in batch 6: `scratchpad/apptest_sample.py` drove the
real app with `streamlit.testing.v1.AppTest`, skipped the landing gate with
`at.session_state["entered"] = True`, clicked the sample button and got a
finished run (no exceptions, 38 findings, four tabs, a labelled sample note) in
about 9 seconds. It needs `default_timeout=420`. This is verified, not proposed.

Open facts that the batches below depend on:

- The run is synchronous inside the Run click handler (`if run_clicked or
  _sample_run:` in `app.py`). There is no Stop and no worker thread.
- `controller.py` has `on_step_callback` and `on_iteration_callback` but no stop
  hook (`grep should_stop src` is empty).
- Findings carry `evidence["checks"]`; Answers cards and the HTML report render
  the check row and a "N of M findings held up" banner already.
- `.streamlit/config.toml` still has the old cream hexes and cites the deleted
  `DESIGN.md`. Streamlit 1.57 supports `[theme.light]` and `[theme.dark]`
  (verified in `streamlit/config.py`).
- `ui/styles.py` loads fonts from Google by `@import`; `static/fonts/` and
  `ledger-fonts.css` exist unused.
- The report (`src/core/html_report.py`) loads fonts and Vega from CDNs;
  `static/vendor/vega/` exists unused.
- `ui/animations.py` is live (called at the bottom of `app.py`) and loads
  Anime.js from a CDN.

## 2. Batch 0: before-screenshots and a clean base

What the user sees: nothing changes. This batch protects the "before" half of
the report's before-and-after screenshots, which the overhaul plan counts as a
Must (2d). It has to happen before batch 2, because the theme-file fix changes
how every native widget looks.

Steps:

1. Commit the pending plan edits and this file on `frontend-overhaul`
   (message: `docs: frontend implementation plan, record decisions`). Owner's
   `README.md` stays out.
2. `git tag batch-0-before-visuals` on that commit. Anyone can later check it
   out to see the exact "before" state.
3. Capture screenshots into `docs/screenshots/before/` at 1440 by 900, Day and
   Night where the theme applies. Needed states:
   - landing page (top and the pricing/FAQ area);
   - empty workspace;
   - file loaded, before running;
   - sample run result: Answers, Charts, Details, Downloads;
   - the exported `report.html` opened from a Downloads button;
   - the plate while idle.
   Name them `NN-state-theme.png`. If the Chrome extension is not connected, the
   owner captures them by hand; keep each file small (a few hundred KB).
4. Commit the screenshots separately (`docs: before screenshots`).

Acceptance: the tag exists; `docs/screenshots/before/` holds one file per state
above; `git status` is clean apart from `README.md`.

Risks: screenshots drifting from the tag if the app changes first. Take them
before starting batch 2.

## 3. Batch 2: results object, "How we got here", answers-first, theme file

What the user sees: the Answers tab opens with the question, the answer and the
checks, then a clearly marked "be careful about" block. Details gains a "How we
got here" section that shows what the system decided and what it had to work
around. Native Streamlit widgets (buttons, tables, inputs) match the page colours
in both themes. This batch carries the viva's architecture story: one typed,
read-only object between the backend and the screens. Overhaul plan references:
Phase 1b B, Phase 4 core, Phase 1 token items.

### 3.1 Step 2a: `RunView` (new file `src/core/run_view.py`)

Purpose: one framework-free, read-only object the UI reads. It adds no analysis.

Where the data comes from, and the one trap. `analysis_decision`,
`degradations`, `unverified_claims`, `deliverable_audit`, `llm_usage` and
`hypothesis_tree` exist only in `agent.memory` contexts, and `agent` is gone
after the rerun. So the builder must be called inside the run handler while
`agent` still exists (inside the worker thread from batch 3 on) and must take a
plain dict snapshot, not a `MemorySystem`. That keeps the module mypy-strict,
free of any Streamlit or memory import, and unit-testable.

Interface (names are a proposal; keep the shape):

```python
RUN_VIEW_CONTEXT_KEYS: tuple[str, ...] = (
    "analysis_decision", "degradations", "unverified_claims",
    "deliverable_audit", "llm_usage", "api_telemetry", "hypothesis_tree",
)

@dataclass(frozen=True)
class Verdict:  held_up: int; audited: int      # needs_more = audited - held_up
@dataclass(frozen=True)
class Decision: mode: str; rationale: str; rejected: tuple[str, ...]
@dataclass(frozen=True)
class Deliverables: delivered: tuple[str, ...]; missing: tuple[str, ...]; repaired: tuple[str, ...]
@dataclass(frozen=True)
class Hypothesis: statement: str; status: str; confidence: float
@dataclass(frozen=True)
class Usage: calls: int; tokens: int; cost_usd: float; is_estimate: bool
@dataclass(frozen=True)
class HowWeGotHere:
    decision: Decision | None
    fallbacks: tuple[str, ...]            # from degradations
    untraced_numbers: tuple[str, ...]     # from unverified_claims
    deliverables: Deliverables | None
    hypotheses: tuple[Hypothesis, ...]
    usage: Usage | None
@dataclass(frozen=True)
class RunView:
    objective: str; is_sample: bool; verdict: Verdict | None
    findings: tuple[dict[str, Any], ...]; coverage: dict[str, Any]
    recommendations: tuple[str, ...]; reasoning: str
    how: HowWeGotHere

def build_run_view(final_result, context, *, objective, is_sample) -> RunView: ...
```

Rules for the builder:

- Absent means absent: a missing context key yields `None` or an empty tuple,
  never a placeholder and never a tick. `Usage` is `None` when `call_count` is 0
  or the run used no LLM.
- `verdict` reuses `audited_checks` from `src/core/audited_entry.py` and the same
  selection the Answers tab builds inline today (the `card_findings` list in
  `render_answers_tab`: layer `exec` or `analyst`, kinds `method_fit` and
  `coverage_gap` excluded, first five). Move that selection into `run_view.py`
  so there is one copy, and expose it as `RunView.headline_findings`.
- Shapes already confirmed in code: `analysis_decision` is a dict with `mode`,
  `target`, `candidate`, `rationale`, `alternatives_rejected`; `degradations` is
  a list of strings (`_add_degradation`); `unverified_claims` is a list of
  strings (`flag_unverified_claims`); `deliverable_audit` has `delivered`,
  `missing`, `repaired`; `llm_usage` has `call_count`, `total_tokens`,
  `estimated_cost_usd`, `is_estimate`; `hypothesis_tree` has `nodes`, a dict of
  `id` to node with `statement`, `status`, `confidence`. Treat every field as
  optional and coerce defensively.
- Cap lists at a small number when building (for example 8 hypotheses, 5
  examples of untraced numbers) and keep the true count.

Wire it up in `app.py`, inside the run handler right after `final = agent.analyze()`:
build the context dict with
`{k: agent.memory.get_context(k) for k in RUN_VIEW_CONTEXT_KEYS}`, call
`build_run_view`, and store the result in `st.session_state["run_view"]`. Add
`run_view` to `_DEFAULTS` and to the keys `_reset_pipeline` clears.

Tests (`tests/test_run_view.py`, pure Python):

- full snapshot: every field populated, counts and caps correct;
- empty snapshot (a no-AI run): `usage is None`, no fallbacks, `how.decision`
  still present when the mode was decided;
- planted case: a refuted hypothesis and a missing deliverable appear;
- absent case: unknown or `None` values never raise and never fabricate;
- verdict equals the count the Answers tab computes today on the same findings.

Add `RunView`-related shape fixtures by hand from the shapes above. Capturing a
real snapshot is better; do it once with the owner's okay by dumping the
`RUN_VIEW_CONTEXT_KEYS` from a CLI no-AI run into `tests/fixtures/`.

Acceptance:

- `ruff check .` and `mypy src/` pass; `tests/test_architecture.py` still passes
  (the module imports nothing from `memory`, `tools`, `engine` or the controller);
- `pytest tests/test_run_view.py -q` passes;
- `git diff --stat src/` shows one new file and no edits to existing backend
  modules.

Commit: `feat: RunView, a read-only results object` (module and tests together).

### 3.2 Step 2b: "How we got here" in Details

New pure builder `ui/components/how_we_got_here.py` with
`build_how_html(how: HowWeGotHere) -> str`. It uses `html.escape` for every
dynamic string, returns `""` when nothing applies, and uses CSS classes only (no
inline `style=`). `ui/tabs/details_tab.py` (function `render_details_tab`)
renders it first, above the "Case Log" heading, and takes the `RunView` as an
argument. Add the classes (a small grid of labelled blocks) to `ui/styles.py`
beside the existing card classes, using `var(--...)` tokens only.

Sections, each hidden when empty, in plain words with a one-line explanation:

1. "What we decided": mode and the rationale; the options it turned down.
2. "What changed along the way": the fallbacks, one line each.
3. "Numbers we could not trace": the count, up to five examples, and a sentence
   saying these are marked and should not be trusted without a check.
4. "What you asked for": delivered, missing, repaired.
5. "Ideas we tested": supported, refuted and inconclusive counts, with the
   statements of refuted and supported ones.
6. "What it cost": AI calls, tokens, estimated cost, labelled as an estimate;
   for a no-AI run, "No AI was used in this run."

Copy rule (overhaul plan section 11): every technical term has a plain
companion; no emoji; sentence case.

Tests (`tests/test_how_we_got_here.py`): hostile text in a rationale and in a
fallback is escaped; empty sections are omitted; a no-AI view shows the "No AI
was used" line and no cost; output contains no `style=`.

Acceptance: `pytest tests/test_how_we_got_here.py -q` passes; manual check (with
the owner's okay) that Details shows the section for the sample run.

Commit: `feat(ui): "How we got here" section in Details`.

### 3.3 Step 2c: answers-first ordering

Target order on the Answers tab (`render_answers_tab` in
`ui/tabs/answers_tab.py`), following the overhaul plan's hierarchy of answer,
evidence, caveat, next action:

1. The question as the heading, with the one-paragraph answer (`reasoning`)
   under it. Take the objective from `RunView`, not from a parameter.
2. The verdict banner ("N of M findings held up"), only when `verdict` exists.
3. "What we found": the finding cards with their check rows (evidence).
4. "Be careful about": one block for caveats: the model stamp
   (`render_defect_stamp`, shown when the train-test gap is risky), the
   unanswered-questions expander, the small-group note, and the
   data-understanding card ("How we read your data", currently near the top).
5. "What to do": the recommendations.
6. "At a glance" (the quality, rows and caveat strip), the run-comparison
   expander (`render_run_compare`), then "Search these findings".
7. "Show technical detail" stays last and collapsed.

Also: drop the `st.info(f"**What we found:** ...")` summary line (the
`dominant` tool block) when finding cards exist, since it repeats the first
card, and keep the fallback to `insights` for runs without findings.

Change the function to read findings, verdict, coverage and recommendations
from `RunView`; keep `tool_results` for the technical expander only until
batch 6. Remove the `count-up` classes and `data-value` attributes in the
expander (see 3.5).

Tests: none new in the pure layer (the order is presentational). Batch 6's
AppTest smoke test asserts the headings appear in this order. Until then,
acceptance is a manual look at the sample run.

Commit: `feat(ui): answers-first ordering on the Answers tab`.

### 3.4 Step 2d: theme file

Problem: `.streamlit/config.toml` carries the old cream and terracotta values and
a stale comment, so native widgets disagree with the CSS in `ui/styles.py`.

Steps:

1. Keep the existing non-colour keys (`font`, `headingFont`, `codeFont`,
   `baseRadius`, `buttonRadius`, `chartCategoricalColors`).
2. Add `[theme.light]` and `[theme.dark]` blocks with `primaryColor`,
   `backgroundColor`, `secondaryBackgroundColor`, `textColor`,
   `dataframeBorderColor`, `dataframeHeaderBackgroundColor`, taken from
   `src/core/design_tokens.py`: pen, stock, sheet_alt, ink, rule, sheet_alt.
3. Replace the header comment: say the TOML is a generated copy of
   `design_tokens.py` and name the script. Remove the reference to `DESIGN.md`.
4. Add `scripts/sync_streamlit_theme.py`: reads `palette("day")` and
   `palette("night")` and rewrites only those keys, so a palette change is one
   command.
5. Add `tests/test_theme_sync.py`: parse the TOML with `tomllib` and assert each
   mapped key equals the token for its mode. A palette edit that forgets the
   script then fails a test.
6. `[server] enableStaticServing = true` is added in batch 5, not here.

Acceptance: `pytest tests/test_theme_sync.py -q` passes; manual check in both
themes that buttons, tables and inputs match the page. Compare against the batch
0 screenshots.

Commit: `fix(ui): sync the Streamlit theme file with the design tokens`.

### 3.5 Step 2e: remove `ui/animations.py`

What it does today: loads Anime.js from a CDN and attaches a `MutationObserver`
to the page on every rerun to animate counters, gauges, handoff items and chart
cards. Checked before planning this step: the markup already contains the final
values (the gauge ring's offset and the counter text are written server-side),
and no CSS hides these elements until the script runs, so removing the script
leaves everything visible. The check-row entrance is pure CSS (`.check-row.animate`
under `prefers-reduced-motion`) and stays; it is the plan's one motion moment.
No test or script references `ui.animations`.

Steps:

1. Search for importers (`grep -rn "ui.animations\|inject_micro" app.py ui src
   tests scripts`); expect only `app.py`.
2. Delete the "MICRO-INTERACTIONS" block at the bottom of `app.py` (the import
   and the call) and delete `ui/animations.py`.
3. In `ui/tabs/answers_tab.py` remove the now-meaningless `count-up` classes and
   `data-value` and `data-suffix` attributes; leave the static ring and numbers.

Acceptance: `grep -rn "anime" app.py ui/*.py ui/tabs ui/components` returns no
Anime.js load from these files (the plate, cinematic export and landing still
have their own; those are "Later"); `ruff check .` passes; manual check that
Answers and the idle page look unchanged apart from the lack of count-up.

Commit: `refactor(ui): remove the CDN animation script`.

### 3.6 Batch 2 acceptance and risks

Acceptance for the batch: all five commits present; the three new test files pass
along with `tests/test_architecture.py`; a manual sample run (owner's okay) shows
the new Answers order, the "How we got here" section and matching widget colours
in both themes; `git diff --stat src/` shows only `run_view.py` as new code.

Risks:

- A wrong guess about a context shape. Mitigation: coerce defensively and test
  with a real captured snapshot.
- The Answers reorder hides the model stamp from people who expand nothing.
  Mitigation: the stamp moves into "Be careful about", which is always visible.
- Theme change affects dataframes and widgets everywhere. Mitigation: compare to
  the batch 0 screenshots; the tag allows a one-step revert.

## 4. Batch 3: run in a worker thread, with Stop

What the user sees: the page stays responsive during a run, a Stop button ends
it early and still shows what finished, and (second half) findings appear as they
are found, clearly marked as provisional. Overhaul plan references: Phase 1b F,
2c item 12, 2b "Live findings: option B".

Steps:

1. Backend stop hook (`src/core/controller.py`). Add `should_stop` (a zero-arg
   callable, default `None`) and a `_stop_requested()` helper. Check it before
   each LLM call and after each step batch in the loop that lives in `_analyze`,
   and between steps inside `_execute_steps`. On stop, return the existing
   deterministic synthesis with `stopped=True` so a partial report is still
   written. This must not change decomposition; if a checkpoint would, stop and
   ask.
2. Offline test (`tests/test_controller_stop.py`): a controller with
   `should_stop=lambda: True` and the LLM off returns a result with a stopped
   marker and does not raise.
3. Run object in `app.py` (move to `ui/run.py` in batch 6): a small `_Run` class
   holding a `threading.Event` for stop, a lock, the stage log and progress
   lines. The worker thread constructs `AgentController` and calls
   `load_dataset` and `analyze`; it never calls any `st.*` function and never
   touches `st.session_state`. The settings it needs travel in a frozen spec,
   as the key, objective and output directory already do. Because the worker
   builds the controller, the objective scope from batch 1 applies on that
   thread automatically.
4. Poll with `@st.fragment(run_every="1s")`: it redraws the stage list and the
   Stop button, and when the worker has finished it folds the outcome into
   `st.session_state` (including `run_view`) and calls `st.rerun()`.
5. Stop sets the event; the controller stops at its next checkpoint; the report
   is built from partial results and the page shows "Stopped at step N".
6. "New analysis" during a run sets a discard flag, signals stop, and resets
   once the worker ends, so no second run can overlap it.
7. Provisional findings (option B). Add an optional `on_finding_callback` to the
   controller, fired where new findings are added to memory (anchor:
   `self.memory.add_findings(new_findings)` in the step-result handling). The
   worker appends the headline, kind and id to the `_Run` under its lock; the
   fragment lists them under "Found so far, may change". These carry no check
   row, because checks, the run-level correction and the audits are attached
   after the loop. `RunView.provisional_findings` is empty once the run
   completes.

Tests: the stop test above; a unit test that `on_finding_callback` receives a
finding when a step adds one; a test that no provisional entry carries a check
mark. AppTest does not drive fragment polling reliably, so the thread behaviour
is checked by hand.

Acceptance: stop returns a result and a report; the page reacts while a run is
in progress; two simultaneous runs in two browser tabs finish with their own
questions and results (manual, owner's okay); `ruff check .` and `mypy src/`
pass.

Commits: stop hook and test; worker and fragment; provisional findings.

Risks: Streamlit rejects `st.*` calls from a worker thread, so keep all UI calls
in the script and fragment; a stuck worker if a tool never returns (Stop is
cooperative only, state this in the UI text); an earlier version of this change
was reverted by the owner because it made the UI uncertain, so land it as small
commits and test each in the browser.

## 5. Batch 4: landing page copy fixes

What the user sees: an entry page that says only what the product can prove, with
one primary action and a plain statement of what happens to uploaded data. The
layout and the 3D scene stay.

Steps:

1. Remove the duplicate prototype: re-point `tests/test_landing_v2.py` from
   `frontend-landing/` to `ui/landing_component/`, confirm with a search that
   nothing else references `frontend-landing/`, delete the directory. Own commit.
2. In `ui/landing_component/index.html` change text and markup only:
   - delete the pricing or "tiers" section and the FAQ section, and the FAQ
     structured-data node if one remains;
   - rewrite the jargon and unsupported claims listed in overhaul plan section 6
     ("swarm synapses", "concentric manifold", "certified safe", exact guarantee
     percentages, fake status chrome);
   - sentence case for headings;
   - add a short data-path statement: the file is processed on the server for
     the session and removed when it ends; summaries of the data go to the
     chosen AI provider unless AI is switched off; this is an academic project
     and not for sensitive data. No retention or compliance promises.
3. Keep the Streamlit bridge (`enter` and `theme` messages) untouched.
4. Run `pytest tests/test_landing.py tests/test_landing_v2.py -q` and update any
   assertion that expects removed sections.

Acceptance: a search of the HTML for the removed phrases returns nothing; the
landing tests pass; manual check that the CTA still enters the workspace and
the theme still carries over.

Risks: these tests may assert removed copy; update them to assert the new
statement instead of deleting them.

## 6. Batch 5: fonts, offline report, fewer cards

What the user sees: no flash of fallback font, a report that opens without the
internet and prints cleanly, and a calmer page with fewer identical boxes.

Steps:

1. Console fonts. In `ui/styles.py` replace the Google `@import` with the local
   `ledger-fonts.css` served from `/app/static/fonts/`, honouring
   `server.baseUrlPath`. Add `[server] enableStaticServing = true` to the config.
   Acceptance: `grep -c "fonts.googleapis" ui/styles.py` is 0 and the network
   tab shows no Google request from the console.
2. Report. In `src/core/html_report.py` read the vendored Vega bundles and the
   Latin-subset fonts once at import and inline them (scripts and data URIs),
   with the existing CDN tags as the fallback if `static/` is missing. Give each
   chart container a server-rendered data table that the script replaces, so
   the page is useful with scripts off. Record the report's byte size and keep
   it under a budget the owner agrees. Acceptance: opened with the network off,
   the charts draw; with scripts off, the tables show; print preview has no
   interactive chrome.
3. Fewer cards. CSS only, in `ui/styles.py`: find the rules that give the same
   border, radius and shadow to the stat tiles, KPI tiles, agent cards and
   panels; keep elevation for the verdict, the plate and expanders; make stat
   tiles figures on a hairline; remove hover lift from elements that are not
   clickable. Acceptance: no static card lifts on hover; compare with batch 0.

Tests: a unit test that the report contains no `fonts.googleapis.com` and no
`cdn.jsdelivr.net` when `static/` is present, and still contains them when it is
not (use a temporary path override); the existing report tests must pass.

Commits: one per step.

Risks: the inlined Vega makes every report about a megabyte or more; the font
change can shift layout, so compare screenshots.

## 7. Batch 6: structure and tests

What the user sees: nothing new. This is the architecture story for the viva:
small modules with one job each, and tests that render the real screens.

Steps:

1. Split `app.py` into `ui/sidebar.py` (returns a frozen settings object, which
   becomes the `RunConfig` input), `ui/run.py` (the worker, the run object, the
   polling fragment), `ui/idle.py` (hero, upload, empty state, sample button),
   `ui/results.py` (the tab host) and `ui/state.py` (the typed session keys and
   `_reset_pipeline`). `app.py` becomes a thin entry. One module per commit,
   moving code without changing it.
2. Split `ui/components/cards.py` into `ui/components/html_builders.py` (pure
   string builders, no `st.*`, no `session_state`) and renderers that call
   them. Replace the remaining inline `style=` in `render_finding_card` with
   classes.
3. Make every tab take the `RunView` and stop calling `find_tool`; keep
   `tool_results` only inside the technical expander and Details tables until
   those are moved onto the view too.
4. UI tests from the verified AppTest harness: copy
   `scratchpad/apptest_sample.py` into `tests/test_app_smoke.py` as a test with
   `default_timeout=420`; assert no exceptions, `analysis_done`, four tabs, the
   sample banner, and the heading order from batch 2. Mark it slow.
5. Session isolation: a test with two `objective_scope` threads and the real
   tool thread pool (the forced-overlap check already written once in the review
   scratch work), and a test that two `AppTest` instances with different
   objectives show only their own.

Acceptance: `app.py` is a short entry file; no module in `ui/` is over roughly
500 lines; `ruff check .`, `mypy src/` and the new tests pass; a manual sample
run behaves as before.

Risks: moving code can silently break a name that a closure captured; move one
module at a time and run the smoke test after each.

## 8. Later (not planned in detail; state as future work in the report)

One shared Three.js runtime across landing, plate and cinematic export, native
scroll for the cinematic export, a full landing rebuild, performance tiers on
real phones, the full accessibility matrix, `analyze_iter` as a full event
generator, and Charts-tab filters. Overhaul plan sections 7, 9 and 11 hold the
specification when time allows.

## 9. Order and dependencies

```text
batch 0 (screenshots) -> batch 2 (RunView, Details, Answers, theme, animations)
                           |-> batch 3 (needs RunView for provisional findings)
                           |-> batch 4 (independent; can run any time after 0)
                           |-> batch 5 (theme step before fonts; after 2d)
batch 6 (needs 2 and 3; finishes the structure story)
```

If time is short, cut from the end: batch 6 steps 4 and 5 are the cheapest part
of it to keep, batch 5 step 3 is the cheapest to drop, and batch 3's provisional
findings are the cheapest part of that batch to drop.

## 10. Status log

| Date | Entry |
|---|---|
| 2026-10-02 | File written. Batches 0 and 2 specified in detail; 3 to 6 specified at step level. Nothing implemented beyond section 1. |
