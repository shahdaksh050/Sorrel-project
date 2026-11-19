# Chart Labeling & Naming Improvements Implementation Plan [COMPLETED & VERIFIED]

> **Status:** All tasks implemented, tested, and verified across all archetypes.
> - `ruff check .`: 0 violations
> - `mypy src/`: 0 errors across 116 source files
> - Unit tests: 71/71 passed (`tests/test_chart_theme.py`, `tests/test_dashboard.py`, `tests/test_clustering.py`, `tests/test_dimensionality_gpu.py`, `tests/test_time_of_day.py`)
> - Chart lint: 341/341 passed (`tests/test_chart_lint.py`)

**Goal:** Eliminate cryptic abbreviations (such as "PC1/2/3", raw snake_case model identifiers, raw integer hours, and sensor codes) across all charts, replacing them with universal, archetype-compatible human-readable labels.

**Architecture:** Enhance the central labeling engine (`src/core/chart_theme.py:humanize_label`), enrich PCA/dimensionality charts (`pca_scree` and `cluster_scatter`) with clear Component titles, format diurnal time-of-day charts with clock hours, and humanize model comparison titles.

**Tech Stack:** Python 3.13, Vega-Lite 5, Pandas, NumPy, scikit-learn.

---

### Task 1: Label Humanizer Expansion (`src/core/chart_theme.py`)

**Files:**
- Modify: `src/core/chart_theme.py:180-215`
- Test: `tests/test_chart_lint.py`

**Step 1: Write failing test for enhanced label humanization**
Assert that `"PC1"` becomes `"Principal Component 1"`, `"pm2.5"` becomes `"Particulate Matter (PM2.5)"`, `"PT08.S1(CO)"` becomes `"CO Sensor (PT08.S1)"`, and `"random_forest"` becomes `"Random Forest"`.

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_chart_lint.py -k test_humanize_label -v`

**Step 3: Implement minimal code in `src/core/chart_theme.py`**
Add regex patterns and dictionary lookups to `humanize_label`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_chart_lint.py -k test_humanize_label -v`

---

### Task 2: PCA Scree & Cluster Scatter Enrichment (`src/core/dashboard_charts_basic.py`)

**Files:**
- Modify: `src/core/dashboard_charts_basic.py:720-795, 885-935`
- Test: `tests/test_dashboard.py`

**Step 1: Write test for enriched PCA component and cluster scatter titles**
Assert that `_scree_chart` and `_cluster_chart` output `"Component 1"`, `"Component 2"` and clear variance titles instead of unadorned `"PC1"` / `"PC2"`.

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -k test_pca -v`

**Step 3: Implement minimal code in `dashboard_charts_basic.py`**
Update `_scree_chart` values generator and `_cluster_chart` axis titles.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -k test_pca -v`

---

### Task 3: Time of Day Diurnal Formatters (`src/tools/time_series.py`)

**Files:**
- Modify: `src/tools/time_series.py:490-508`
- Test: `tests/test_time_series.py`

**Step 1: Write test for formatted hour labels**
Assert that `_tod_charts` generates clock hour labels (`12 AM`, `1 AM`... `12 PM`, `1 PM`...) instead of raw integers.

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_time_series.py -k test_tod -v`

**Step 3: Implement minimal code in `time_series.py`**
Format hours with `_format_hour_label(h)`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_time_series.py -k test_tod -v`

---

### Task 4: Model Comparison & Feature Importance Names (`src/core/dashboard_charts_model.py`)

**Files:**
- Modify: `src/core/dashboard_charts_model.py:30-65, 100-130`
- Test: `tests/test_dashboard.py`

**Step 1: Write test for humanized model names**
Assert that model comparison chart displays `"Random Forest"`, `"XGBoost"`, `"Logistic Regression"` instead of snake_case identifiers.

**Step 2: Run test and verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -k test_model_comparison -v`
