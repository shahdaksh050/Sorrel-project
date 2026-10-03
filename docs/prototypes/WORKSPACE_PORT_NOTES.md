# Workspace prototype: port notes

The Workspace surface in `verdacert_dsa_preview.html` (`#surface-workspace`) is the
original Streamlit workspace's structure and rules in the Sorrel look. Nothing in
`app.py` or `ui/` was changed. This file says what the prototype keeps, what it
changes, and what a port has to do.

## What the prototype keeps from the original workspace

| Original (app.py, ui/tabs) | Prototype |
| :--- | :--- |
| Empty state: "Start with the data." and the three-step promise | Same headline and sub-line |
| Steps 1 to 4: Add your file, Ask your question, Choose how to answer, Run | Same four steps, same labels and captions |
| `Use sample data` button, "not your data" | Same, but the sample is the Air Quality run (see below) |
| Run gating: needs a key unless "Without an AI summary" | Same, including "Run without an AI summary instead" |
| Dataset preview: Rows, Columns, Missing cells, Numeric columns, first 10 rows, column types | Same, built from the real file through `src.core.io.read_any_bytes` |
| File notices (auto-repaired) | Same list, rewritten in plain words |
| Running: seven named stages (`STAGE_DEFS`), Stop, a layout-stable skeleton | Same names; timeline plus a static skeleton shaped like Answers |
| Failed: "The analysis did not finish..." and Technical details | Same message; the raw error text is described, not invented |
| Datum line: State, File, Analysis | Same three cells |
| Settings sidebar: Connect an AI, AI and model options, Advanced analysis, Safety | Same controls and help text, in a rail opened by a Settings button |
| Tabs: Answers, Charts, Details, Downloads | Same four, as a real tablist with an underline for the active tab |
| Answers order: summary, "N of M checked findings held up", What we found (ranked list plus selected finding), Be careful about, What to do, At a glance, Search | Same order and wording |
| Check row marks: tick, exclamation, circle (shape plus words) | Same |

## What the prototype changes

- Inputs live in the main area as one calm task card. The settings rail holds only settings.
- Technical figures sit in folded "Details for analysts" notes; the plain sentence leads.
- `--risk`/`--danger` marks only "this may not hold". The Failed state uses amber.
- No gradients, glow, transitions or animation anywhere in the workspace CSS.
- No 3D in the workspace (the original's optional "3D progress view" toggle is left out).
- Details is an audit trail: "How we got here", "Your data and the checks", "Safety and audit", then a folded step record.
- Downloads is an artifact shelf with a type tag, a purpose, a size and an availability state.

## Data behind the demo

Every number comes from `output/aq_final` (the real run) or from reading `data/AirQualityUCI.csv`
with the app's own reader. Static checks live in `docs/prototypes/checks/` and are run by hand:

```
PYTHONIOENCODING=utf-8 python docs/prototypes/checks/check_workspace.py   # structure, ARIA, wording, number traceability
PYTHONIOENCODING=utf-8 python docs/prototypes/checks/check_landing.py     # the landing surface
PYTHONIOENCODING=utf-8 python docs/prototypes/checks/contrast.py          # AA ratios, day and night
```

They are one-off prototype checks, not part of `pytest`, and nothing in the page was looked at in a browser.

- The pair figures (7,344 readings, r = 0.9311, slope 4.82, 2,127 left out, p rounds to 0) are **not stored in the run's files**. They were re-derived from the data with the run's own cleaning rule (-200 = missing). The run's stored finding is the group result "r between 0.88 and 0.98".
- The run had no objective (it was an autonomous, no-AI run), so the question box in the mock is empty. Any question shown would be an invented example.
- The prototype shows three findings chosen by hand (correlation, weak groups, a level change); the real app ranks them from `RunView.headline_findings`.
- The "Search the written findings" box searches 12 plain sentences, one per real insight. Each keeps only its own insight's numbers; r and the silhouette score stay in the folded notes, and the original codes are kept as hidden search aliases.
- Night mode: the raw `--accent` and `--danger` fall under AA as text, so text uses the derived `--accent-text` and `--danger-text` tokens (day values are unchanged). This also fixed pre-existing landing text (verdict banner, pills, risk checks) in night mode.
- The sample button here loads Air Quality. The real app's button loads `data/sample_customer_churn.csv`. Decide which sample ships before porting.

## Port checklist (when the time comes)

1. Port the Sorrel tokens into `ui/styles.py` (day and night) and update `tests/test_theme_sync.py` and `tests/test_islands.py`, which assert the Ledger palette and contrast.
2. Rebuild the four steps and the settings groups with Streamlit widgets in the same order; keep every existing widget key.
3. Render the check rows, banner and "Details for analysts" folds from `RunView` (`ui/tabs/answers_tab.py`); an absent check renders nothing, never a tick.
4. Charts tab: pair every chart with a takeaway, the check row and a provenance line. The schematic charts in the prototype are placeholders for the real chart data.
5. Details: add "How we got here" from `RunView.how` (mode decision and rejected options, fallbacks, unverified numbers, outputs delivered, usage footer when an LLM ran).
6. Keep all interpolated text `html.escape`d. Keep the tablist keyboard behaviour (arrows, Home, End) if tabs are rebuilt outside `st.tabs`.
7. Left out on purpose and worth deciding: descriptive statistics table in the preview, "Run compare", the 3D progress view toggle.
