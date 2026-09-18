"""
Planted-effects dataset corpus — IMPROVEMENTS.md 7.11.

Where tests/fixtures/__init__.py generates *shape* edge cases (malformed
files, weird dtypes, degenerate row/column counts) to prove the pipeline
fails loud or succeeds cleanly, this module generates *substantively
correct* datasets with known, documented effects baked in — so a test can
run the real pipeline end to end and check whether it actually recovers
the thing that's really there, not just that it doesn't crash.

Every generator:
  - takes `path` (a `Path`, typically pytest's `tmp_path / "name.csv"`) and
    writes a CSV there, returning the `Path` — or, if `path` is None,
    returns the `pandas.DataFrame` directly without touching disk. This
    mirrors tests/fixtures/__init__.py's `(tmp_path) -> Path` convention,
    extended with the DataFrame-only escape hatch these tests also want.
  - takes `n` (row count) and `seed` (int, fed to `np.random.default_rng`)
    for reproducibility — same seed, same file, every run.
  - returns `(csv_path_or_dataframe, plants)` where `plants` is a plain
    dict documenting exactly what was planted: the columns/levels involved
    and the exact magnitude used to generate the data (not a rounded or
    aspirational number — the literal constant the generator used).

`plants["checks"]` is a uniform sub-list every generator populates, each
entry shaped `{"kind": ..., "criteria": {...}, "effect": ..., "effect_tol":
...}` — this is what tests/test_planted_effects.py's `score_recovery()`
iterates over to compute a recovered/total score without needing
generator-specific logic. The rest of `plants` is free-form and exists so
a human can check the magnitude independently of re-reading this file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _finalize(df: pd.DataFrame, path: Path | None) -> Path | pd.DataFrame:
    """Write `df` to `path` and return the Path, or return `df` untouched
    when no path is given — the calling convention every generator below
    shares."""
    if path is None:
        return df
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. Transactional — region premium, seasonal lift, repeat-purchase structure
# ---------------------------------------------------------------------------

def generate_transactional(
    path: Path | None = None, n: int = 4000, seed: int = 0
) -> tuple[Path | pd.DataFrame, dict[str, Any]]:
    """Order-line transactions with three planted, independent effects:

    1. A +25% revenue premium in `region == "West"` vs every other region
       on the `amount` column (segment_lift, ratio 1.25x / lift 0.25) —
       this is the literal example in segment_comparison.py's own
       docstring ("West-region orders run 25% above the company average").
    2. A x1.4 seasonal lift on `amount` in November and December vs the
       yearly average (the "change" / calendar-month-factor finding in
       time_series.py).
    3. ~10 rows per `customer_id` (exact, by construction) — repeat-
       purchase structure that both trips the TRANSACTIONAL domain's
       structural score (src/core/domains.py `_transactional_structure`)
       and gives cohort_analysis.py's concentration/repeat-rate findings
       something real to measure.
    """
    rng = np.random.default_rng(seed)

    n_customers = max(1, n // 10)  # exactly ~10 rows/customer by construction
    customer_idx = np.arange(n) % n_customers
    rng.shuffle(customer_idx)
    customer_id = [f"CUST-{i:05d}" for i in customer_idx]

    order_id = [f"ORD-{i:06d}" for i in range(n)]
    region = rng.choice(["North", "South", "East", "West"], size=n)
    product_category = rng.choice(
        ["Electronics", "Apparel", "Home", "Grocery", "Toys"], size=n
    )

    # Two full years so both November and December occur repeatedly.
    order_date = pd.to_datetime(
        rng.integers(0, 730, size=n), unit="D", origin="2022-01-01"
    )

    unit_price = rng.gamma(shape=9.0, scale=6.0, size=n)  # mean ~54
    quantity = rng.integers(1, 6, size=n)
    base_amount = unit_price * quantity

    REGION_PREMIUM = 0.25          # West runs 25% above the rest (ratio 1.25x)
    region_mult = np.where(region == "West", 1.0 + REGION_PREMIUM, 1.0)

    SEASONAL_LIFT = 0.4            # Nov/Dec run 40% above the rest (ratio 1.4x)
    month = order_date.month
    season_mult = np.where(month.isin([11, 12]), 1.0 + SEASONAL_LIFT, 1.0)

    # A genuine top-decile revenue concentration among customers. Exactly-
    # equal order counts per customer (the repeat-purchase plant above) is
    # NOT, by itself, a concentration story: with ~10 i.i.d. line items per
    # customer, the top 10% of customers land around 13-14% of revenue by
    # CLT alone — under cohort_analysis.py's and concentration_analysis.py's
    # shared 15%-share triviality floor (T5), so no concentration finding
    # would ever fire. VIP_MULT gives exactly the top decile of customer IDs
    # (by construction) a real per-line-item spend premium so the top 10%
    # of customers genuinely drive ~25% of revenue — comfortably above the
    # 15% floor.
    N_VIP_CUSTOMERS = max(1, n_customers // 10)   # exactly the top decile
    VIP_MULT = 3.0                                 # VIP customers spend 3x per line item
    is_vip_customer = customer_idx < N_VIP_CUSTOMERS
    vip_mult = np.where(is_vip_customer, VIP_MULT, 1.0)

    noise = rng.normal(1.0, 0.05, size=n)  # 5% multiplicative noise, realism only
    discount_pct = rng.uniform(0.0, 0.2, size=n)

    amount = base_amount * region_mult * season_mult * vip_mult * noise * (1 - discount_pct)
    amount = np.round(np.clip(amount, 1.0, None), 2)

    df = pd.DataFrame(
        {
            "order_id": order_id,
            "order_date": order_date,
            "customer_id": customer_id,
            "region": region,
            "product_category": product_category,
            "quantity": quantity,
            "amount": amount,
            "discount_pct": np.round(discount_pct, 4),
        }
    )

    plants: dict[str, Any] = {
        "region_premium": {
            "dimension": "region",
            "level": "West",
            "measure": "amount",
            "lift": REGION_PREMIUM,   # 0.25
            "ratio": 1.0 + REGION_PREMIUM,
        },
        "seasonal_lift": {
            "dimension": "month",
            "levels": ["November", "December"],
            "measure": "amount",
            "lift": SEASONAL_LIFT,    # 0.4
            "ratio": 1.0 + SEASONAL_LIFT,
        },
        "repeat_purchase": {
            "entity_column": "customer_id",
            "avg_rows_per_customer": n / n_customers,  # exactly 10.0 for n=4000
            "n_customers": n_customers,
        },
        "customer_concentration": {
            "dimension": "customer",
            "measure": "revenue_share",
            "n_vip_customers": N_VIP_CUSTOMERS,   # exactly the top decile
            "vip_mult": VIP_MULT,                 # 3.0x per line item
            "note": (
                "top decile of customers spends 3x per line item, so they "
                "drive well over the 15%-share triviality floor shared by "
                "cohort_analysis.py and concentration_analysis.py"
            ),
        },
        "checks": [
            {
                "kind": "segment_lift",
                "criteria": {"dimension": "region", "level": "West", "measure": "amount"},
                "effect": REGION_PREMIUM,
                "effect_tol": 0.35,
            },
            {
                "kind": "change",
                "criteria": {
                    "dimension": "month",
                    "level": ["November", "December"],
                    "measure": "amount",
                },
                "effect": SEASONAL_LIFT,
                "effect_tol": 0.4,
            },
            {
                "kind": "concentration",
                "criteria": {"dimension": "customer", "measure": "revenue_share"},
            },
            {
                "kind": "cohort",
                "criteria": {"dimension": "customer", "measure": "repeat_purchase_rate"},
            },
        ],
    }
    return _finalize(df, path), plants


# ---------------------------------------------------------------------------
# 2. Churn — dominant categorical driver + weak numeric driver + pure noise
# ---------------------------------------------------------------------------

def generate_churn(
    path: Path | None = None, n: int = 2000, seed: int = 1
) -> tuple[Path | pd.DataFrame, dict[str, Any]]:
    """Churn dataset with a dominant categorical driver (`contract`) and a
    weak numeric driver (`monthly_charges`), plus pure-noise columns.

    `contract == "month-to-month"` churns at exactly 3x the baseline rate
    shared by `one-year` and `two-year` (0.36 vs 0.12 — this is also
    segment_comparison.py's own docstring example almost verbatim:
    "Month-to-month customers churn at 42% vs a 27% baseline"). Any driver
    or segment-lift finding on `monthly_charges` should be much weaker than
    the one on `contract`.
    """
    rng = np.random.default_rng(seed)

    BASE_RATE = 0.12                 # one-year and two-year churn rate
    MONTH_TO_MONTH_MULT = 3.0        # documented multiple of the base rate
    MONTH_TO_MONTH_RATE = BASE_RATE * MONTH_TO_MONTH_MULT  # 0.36
    CONTRACT_PROBS = {"month-to-month": 0.5, "one-year": 0.3, "two-year": 0.2}

    contract = rng.choice(
        list(CONTRACT_PROBS.keys()), size=n, p=list(CONTRACT_PROBS.values())
    )
    base_p = np.where(contract == "month-to-month", MONTH_TO_MONTH_RATE, BASE_RATE)

    monthly_charges = rng.normal(70.0, 20.0, size=n)
    # Weak numeric nudge: a $1 move in monthly_charges shifts churn
    # probability by ~0.1pp — barely above noise, deliberately not a
    # usable standalone driver.
    numeric_nudge = (monthly_charges - 70.0) / 20.0 * 0.02
    p = np.clip(base_p + numeric_nudge, 0.01, 0.95)
    churn = (rng.random(n) < p).astype(int)

    tenure_months = rng.integers(1, 72, size=n)
    customer_id = [f"CUST-{i:05d}" for i in range(n)]
    noise_a = rng.normal(0, 1, size=n)
    noise_b = rng.choice(["X", "Y", "Z"], size=n)
    noise_c = rng.uniform(0, 1, size=n)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "contract": contract,
            "tenure_months": tenure_months,
            "monthly_charges": np.round(monthly_charges, 2),
            "noise_a": np.round(noise_a, 4),
            "noise_b": noise_b,
            "noise_c": np.round(noise_c, 4),
            "churn": churn,
        }
    )

    # segment_comparison.py computes lift against the rest-of-data baseline
    # (the other levels: one-year and two-year), so the tested comparison,
    # ratio, and lift is level vs everyone else (`rest_vals`).
    # MONTH_TO_MONTH_RATE / BASE_RATE - 1.0 is 2.0 (ratio 3.0x).
    _expected_lift_vs_rest = MONTH_TO_MONTH_RATE / BASE_RATE - 1.0

    plants: dict[str, Any] = {
        "dominant_categorical_driver": {
            "column": "contract",
            "level": "month-to-month",
            "target": "churn",
            "base_rate": BASE_RATE,             # 0.12
            "level_rate": MONTH_TO_MONTH_RATE,  # 0.36
            "multiple_of_base": MONTH_TO_MONTH_MULT,  # 3.0 (vs the OTHER levels only)
            "expected_lift": _expected_lift_vs_rest,  # 2.0 (vs rest-of-data baseline)
        },
        "weak_numeric_driver": {
            "column": "monthly_charges",
            "note": "barely-above-noise effect on churn probability (~0.1pp per $1)",
            "expected_effect_magnitude": "small — should not out-rank 'contract'",
        },
        "noise_columns": ["noise_a", "noise_b", "noise_c", "tenure_months"],
        "checks": [
            {
                "kind": "segment_lift",
                "criteria": {
                    "dimension": "contract",
                    "level": "month-to-month",
                    "measure": "churn",
                },
                "effect": _expected_lift_vs_rest,  # 2.0 (vs rest-of-data baseline)
                "effect_tol": 0.5,
            },
            {
                "kind": "driver",
                "criteria": {"dimension": "contract", "level": "month-to-month"},
            },
        ],
    }
    return _finalize(df, path), plants


# ---------------------------------------------------------------------------
# 3. Price series — a documented drawdown
# ---------------------------------------------------------------------------

def generate_price_series(
    path: Path | None = None, n: int = 750, seed: int = 2
) -> tuple[Path | pd.DataFrame, dict[str, Any]]:
    """Daily 'price' series with normal drift+noise everywhere except a
    clean, documented peak-to-trough drawdown of exactly `DRAWDOWN_PCT`
    over `DRAWDOWN_DAYS` days starting at `PEAK_IDX`.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    daily_ret = rng.normal(0.0004, 0.009, size=n)
    price = 100.0 * np.cumprod(1.0 + daily_ret)

    PEAK_IDX = 300
    DRAWDOWN_DAYS = 60
    TROUGH_IDX = PEAK_IDX + DRAWDOWN_DAYS
    DRAWDOWN_PCT = 30.0  # documented peak-to-trough decline, in percent

    peak_price = float(price[PEAK_IDX])
    trough_price = peak_price * (1.0 - DRAWDOWN_PCT / 100.0)
    decline_path = np.linspace(peak_price, trough_price, DRAWDOWN_DAYS + 1)
    decline_noise = rng.normal(0.0, peak_price * 0.003, size=DRAWDOWN_DAYS + 1)
    price[PEAK_IDX : TROUGH_IDX + 1] = decline_path + decline_noise

    # Resume the noisy drift from the actual (noisy) trough level, rather
    # than snapping back to the pre-drawdown trajectory.
    resumed = float(price[TROUGH_IDX]) * np.cumprod(1.0 + daily_ret[TROUGH_IDX + 1 :])
    price[TROUGH_IDX + 1 :] = resumed

    # Cap any OTHER peak-to-trough decline in the series (before the plant,
    # or in the post-recovery random walk) at just above the planted
    # magnitude. Without this, the surrounding daily_ret noise (0.9%/day
    # over 750 days) is perfectly capable of producing its own, unrelated
    # decline bigger than the planted one purely by chance — and financial_
    # analysis.py's max_drawdown is (correctly) a *global* peak-to-trough
    # measure, so an unrelated organic dip elsewhere would silently become
    # "the" max drawdown instead of the one this fixture documents. This
    # floor makes the planted drawdown the genuine global maximum by
    # construction, while leaving the plant itself (~30%, comfortably under
    # the cap) untouched.
    _DRAWDOWN_CAP_PCT = DRAWDOWN_PCT + 5.0  # 35% — small margin over the plant
    _running_peak = np.maximum.accumulate(price)
    _floor = _running_peak * (1.0 - _DRAWDOWN_CAP_PCT / 100.0)
    price = np.maximum(price, _floor)

    df = pd.DataFrame({"date": dates, "price": np.round(price, 4)})

    plants: dict[str, Any] = {
        "drawdown": {
            "measure": "max_drawdown",
            "drawdown_pct": DRAWDOWN_PCT,  # 30.0
            "peak_idx": PEAK_IDX,
            "trough_idx": TROUGH_IDX,
            "peak_date": str(dates[PEAK_IDX].date()),
            "trough_date": str(dates[TROUGH_IDX].date()),
            "peak_price": round(peak_price, 4),
            "trough_price": round(trough_price, 4),
        },
        "date_column": "date",
        "price_column": "price",
        "checks": [
            {
                "kind": "financial",
                "criteria": {"measure": "max_drawdown"},
                # Sign convention for max_drawdown_pct is not pinned down
                # in this file — tests should compare abs(evidence value).
            },
        ],
    }
    return _finalize(df, path), plants


# ---------------------------------------------------------------------------
# 4. HR roster — a planted department pay gap
# ---------------------------------------------------------------------------

def generate_hr_roster(
    path: Path | None = None, n: int = 1200, seed: int = 3
) -> tuple[Path | pd.DataFrame, dict[str, Any]]:
    """HR roster with a planted unadjusted median pay gap between the
    lowest-paid department ("Sales") and the highest-paid ("Engineering")
    of ~30% ((Engineering_median - Sales_median) / Engineering_median),
    matching exactly how workforce_analysis.py computes
    `unadjusted_median_pay_gap`.
    """
    rng = np.random.default_rng(seed)

    departments = ["Sales", "Engineering", "Marketing", "Operations", "Finance"]
    dept_p = [0.30, 0.25, 0.15, 0.20, 0.10]
    # Target department salary means. Sales is deliberately the lowest,
    # Engineering the highest; the gap between them is the plant.
    DEPT_MEAN_SALARY = {
        "Sales": 70_000.0,
        "Engineering": 100_000.0,
        "Marketing": 90_000.0,
        "Operations": 85_000.0,
        "Finance": 95_000.0,
    }
    PAY_GAP_PCT = round(
        (DEPT_MEAN_SALARY["Engineering"] - DEPT_MEAN_SALARY["Sales"])
        / DEPT_MEAN_SALARY["Engineering"]
        * 100,
        2,
    )  # 30.0

    department = rng.choice(departments, size=n, p=dept_p)
    salary_mean = np.array([DEPT_MEAN_SALARY[d] for d in department])
    salary = rng.normal(salary_mean, salary_mean * 0.15, size=n)
    salary = np.round(np.clip(salary, 20_000.0, None), 2)

    job_title = rng.choice(
        ["Associate", "Analyst", "Manager", "Senior", "Director"], size=n
    )
    tenure_years = np.round(rng.exponential(scale=4.0, size=n).clip(0.1, 30.0), 2)
    employee_id = [f"EMP-{i:05d}" for i in range(n)]

    df = pd.DataFrame(
        {
            "employee_id": employee_id,
            "department": department,
            "job_title": job_title,
            "tenure_years": tenure_years,
            "salary": salary,
        }
    )

    plants: dict[str, Any] = {
        "pay_gap": {
            "dimension": "department",
            "higher_level": "Engineering",
            "lower_level": "Sales",
            "measure": "median_pay",
            "gap_pct": PAY_GAP_PCT,  # 30.0
            "dept_target_means": DEPT_MEAN_SALARY,
        },
        "checks": [
            {
                "kind": "segment_lift",
                "criteria": {"dimension": "department", "measure": "median_pay"},
                "effect": PAY_GAP_PCT / 100.0,  # 0.30
                "effect_tol": 0.5,
            },
        ],
    }
    return _finalize(df, path), plants


# ---------------------------------------------------------------------------
# 5. Panel — within-group trend + one high-lift entity
# ---------------------------------------------------------------------------

def generate_panel(
    path: Path | None = None, n: int = 1500, seed: int = 4
) -> tuple[Path | pd.DataFrame, dict[str, Any]]:
    """A store/month sales panel: `N_STORES` stores each observed over
    `N_PERIODS` months (`N_STORES * N_PERIODS == n`). Two independent
    plants:

    1. An overall upward trend across periods, shared by every store
       (`TOTAL_GROWTH_PCT` cumulative growth from period 0 to the last
       period).
    2. `Store_00` runs `ENTITY_LIFT` (50%) above every other store's
       baseline level, at every period — a segment_lift dimension/level
       pair, same mechanism as the region premium in the transactional
       fixture.

    `N_STORES` is kept at 15 (under segment_comparison.py's
    `_MAX_DIM_CARD = 20`) so the entity-lift comparison is a real
    "segment" in that tool's sense, not an ID sweep it would refuse.
    """
    rng = np.random.default_rng(seed)

    N_STORES = 15
    N_PERIODS = n // N_STORES  # 100 for n=1500
    assert N_STORES * N_PERIODS == n, "generate_panel: n must be divisible by 15"

    store_ids = [f"Store_{i:02d}" for i in range(N_STORES)]
    dates = pd.date_range("2020-01-01", periods=N_PERIODS, freq="MS")

    base_level = rng.normal(10_000.0, 1_000.0, size=N_STORES)

    TOTAL_GROWTH_PCT = 60.0  # cumulative growth over the full window, in percent
    period_growth = np.linspace(0.0, TOTAL_GROWTH_PCT / 100.0, N_PERIODS)

    OUTLIER_STORE = "Store_00"
    ENTITY_LIFT = 0.5  # Store_00 runs 50% above baseline at every period

    store_idx = np.repeat(np.arange(N_STORES), N_PERIODS)
    period_idx = np.tile(np.arange(N_PERIODS), N_STORES)

    level = base_level[store_idx]
    growth = 1.0 + period_growth[period_idx]
    lift = np.where(np.array(store_ids)[store_idx] == OUTLIER_STORE, 1.0 + ENTITY_LIFT, 1.0)
    noise = rng.normal(1.0, 0.05, size=n)

    sales = level * growth * lift * noise

    df = pd.DataFrame(
        {
            "store_id": np.array(store_ids)[store_idx],
            "month": dates[period_idx],
            "sales": np.round(sales, 2),
        }
    )

    plants: dict[str, Any] = {
        "within_group_trend": {
            "measure": "sales",
            "date_column": "month",
            "total_growth_pct": TOTAL_GROWTH_PCT,  # 60.0
            "n_periods": N_PERIODS,
        },
        "entity_lift": {
            "dimension": "store_id",
            "level": OUTLIER_STORE,
            "measure": "sales",
            "lift": ENTITY_LIFT,  # 0.5
        },
        "n_stores": N_STORES,
        "checks": [
            {
                "kind": "trend",
                "criteria": {"measure": "sales"},
            },
            {
                "kind": "segment_lift",
                "criteria": {
                    "dimension": "store_id",
                    "level": OUTLIER_STORE,
                    "measure": "sales",
                },
                "effect": ENTITY_LIFT,
                "effect_tol": 0.4,
            },
        ],
    }
    return _finalize(df, path), plants


# ---------------------------------------------------------------------------
# 6. Pure description — genuinely no plantable target
# ---------------------------------------------------------------------------

def generate_pure_description(
    path: Path | None = None, n: int = 500, seed: int = 5
) -> tuple[Path | pd.DataFrame, dict[str, Any]]:
    """A warehouse-inspection-style dataset with no column name that hints
    at a prediction target. Column names are deliberately checked against
    src/core/memory.py's `_NUMERIC_TARGET_NAMES` (profit, sales, revenue,
    price, amount, total, cost, score, rating, demand, margin, result,
    status), `_DOMAIN_TARGET_NAMES` (churn, survived, fraud, default,
    purchased, converted, cancelled, canceled, clicked, subscribed,
    is_fraud, is_churn, is_default) and `_PARTIAL_TARGET_HINTS` substrings
    (target, label, class, outcome, predict, response) — none of the
    columns below match any of those, by construction. A correct system
    should decline to auto-model (T2: `analysis_decision["mode"] ==
    "describe"` with `target is None`) rather than force a regression on a
    name-matched column that doesn't exist here.
    """
    rng = np.random.default_rng(seed)

    zone = rng.choice(["A", "B", "C", "D"], size=n)
    material_type = rng.choice(["Plastic", "Metal", "Wood", "Glass"], size=n)
    pallet_weight_kg = np.round(rng.normal(12.5, 3.0, size=n).clip(min=0.1), 2)
    storage_temp_c = np.round(rng.normal(18.0, 4.0, size=n), 2)
    inspected_by = rng.choice(["J. Lee", "M. Ortiz", "A. Chen", "S. Patel"], size=n)
    last_inspection_date = pd.date_range("2023-01-01", periods=n, freq="6h")
    asset_tag = [f"AST-{i:05d}" for i in range(n)]

    df = pd.DataFrame(
        {
            "asset_tag": asset_tag,
            "zone": zone,
            "material_type": material_type,
            "pallet_weight_kg": pallet_weight_kg,
            "storage_temp_c": storage_temp_c,
            "inspected_by": inspected_by,
            "last_inspection_date": last_inspection_date,
        }
    )

    plants: dict[str, Any] = {
        "expected_mode": "describe",
        "expected_target": None,
        "avoided_keyword_sets": [
            "_NUMERIC_TARGET_NAMES",
            "_DOMAIN_TARGET_NAMES",
            "_PARTIAL_TARGET_HINTS",
        ],
        "checks": [],  # no Finding-based checks — this is a mode-decision test
    }
    return _finalize(df, path), plants


# ---------------------------------------------------------------------------
# 7. Text column — a planted sentiment/keyword skew
# ---------------------------------------------------------------------------

def generate_text_column(
    path: Path | None = None, n: int = 800, seed: int = 6
) -> tuple[Path | pd.DataFrame, dict[str, Any]]:
    """A `review_text` free-text column (long enough per-review to clear
    profiler.py's `FREE_TEXT_AVG_WORDS = 6.0` threshold) alongside two
    structured columns. `NEGATIVE_FRACTION` (0.30) of reviews are drawn
    from templates that all contain the literal word "terrible" plus other
    negative vocabulary; positive templates deliberately vary their
    vocabulary so no single positive word reaches a comparable frequency —
    this makes "terrible" the single most frequent non-stopword token,
    which is exactly the mechanism text_analysis.py's `findings()` uses
    (most-frequent-token occurrence rate > 5%).
    """
    rng = np.random.default_rng(seed)

    NEGATIVE_FRACTION = 0.30
    SIGNATURE_KEYWORD = "terrible"

    negative_templates = [
        "This product was terrible and broke after only two days of use",
        "Absolutely terrible quality, I want a refund immediately for this",
        "Terrible experience overall, the item arrived damaged and unusable",
        "I regret this purchase, terrible build quality and awful customer support",
        "Terrible value for the price, would not recommend this to anyone",
        "The packaging was terrible and the product inside was already broken",
        "Terrible design choices made this completely unacceptable for daily use",
        "Such a terrible waste of money, nothing about this worked as promised",
    ]
    positive_templates = [
        "This exceeded every expectation I had and works beautifully every day",
        "Solid craftsmanship and fast shipping made this a delightful purchase overall",
        "I have used this daily for months and it still performs wonderfully",
        "Great value, thoughtful design, and excellent support when I had questions",
        "Very pleased with the finish and the attention to small details here",
        "Reliable, comfortable, and easy to set up right out of the box",
        "A genuinely pleasant surprise given how affordable this turned out to be",
        "Works exactly as described and the build quality feels impressively sturdy",
    ]

    is_negative = rng.random(n) < NEGATIVE_FRACTION
    neg_choice = rng.integers(0, len(negative_templates), size=n)
    pos_choice = rng.integers(0, len(positive_templates), size=n)
    review_text = [
        negative_templates[neg_choice[i]] if is_negative[i] else positive_templates[pos_choice[i]]
        for i in range(n)
    ]

    category = rng.choice(["Electronics", "Home", "Sports", "Beauty"], size=n)
    store_location = rng.choice(["Online", "In-Store"], size=n)
    review_id = np.arange(n)

    df = pd.DataFrame(
        {
            "review_id": review_id,
            "category": category,
            "store_location": store_location,
            "review_text": review_text,
        }
    )

    plants: dict[str, Any] = {
        "text_column": "review_text",
        "negative_fraction": NEGATIVE_FRACTION,  # 0.30
        "signature_keyword": SIGNATURE_KEYWORD,  # "terrible" — present in every negative review
        "n_negative": int(is_negative.sum()),
        "checks": [
            {
                "kind": "text",
                "criteria": {"dimension": "review_text", "level": SIGNATURE_KEYWORD},
                "effect": NEGATIVE_FRACTION,
                "effect_tol": 0.4,
            },
        ],
    }
    return _finalize(df, path), plants
