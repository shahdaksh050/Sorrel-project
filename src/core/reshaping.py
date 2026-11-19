"""
Reshaping & Ingestion Robustness — wide-table and report-layout normalisation.

Detects non-tidy data representations commonly found in government data,
utility reports, and financial exports:
1. Wide "time in header" tables (e.g. columns named '2015', '2016', ... or 'Jan', 'Feb', ...)
   that represent longitudinal panel or time-series data spread across columns.
2. Summary/subtotal rows (e.g. 'Total', 'Grand Total', 'Average') that distort
   distributions and sample size if left inside data rows.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.geo_stats import haversine_distance, spherical_center

if TYPE_CHECKING:
    from src.core.profiler import DatasetArchetype

_YEAR_RE = re.compile(r"^(?:y|year)?(19\d{2}|20\d{2})$", re.IGNORECASE)
_QUARTER_RE = re.compile(r"^(?:q[1-4][-_\s]?\d{2,4}|\d{2,4}[-_\s]?q[1-4])$", re.IGNORECASE)
_MONTH_NAMES = frozenset({
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december",
})

_SUMMARY_TOKENS = frozenset({
    "total", "grand total", "subtotal", "average", "mean", "all", "sum", "overall",
})

_POSITIVE_WORDS = frozenset({
    "good", "great", "excellent", "outstanding", "amazing", "love", "best", "fast",
    "helpful", "smooth", "perfect", "reliable", "satisfied", "positive", "high",
    "quality", "superb", "delight", "pleased", "recommend",
})
_NEGATIVE_WORDS = frozenset({
    "bad", "poor", "terrible", "horrible", "worst", "broken", "slow", "defect",
    "failure", "refund", "clunky", "bug", "error", "fail", "complaint", "negative",
    "disappointed", "flickering", "unreliable", "dissatisfied",
})
_STOPWORDS = frozenset(
    "a an the of in on for to and or by with what which who how why is are was were "
    "do does did be been it its this that these those from at as vs versus than per "
    "across between over under into about my our me i we you show find tell analyse "
    "analyze analysis data dataset any most more less much many should can could would "
    "there their them".split()
)

_SURVIVAL_WORD_MAP: dict[str, float] = {
    **dict.fromkeys(
        ("1", "1.0", "yes", "y", "true", "t", "event", "observed", "died", "dead", "death", "deceased",
         "churned", "churn", "left", "failed", "failure", "converted", "relapsed", "relapse",
         "terminated", "dropped", "dropout", "attrited", "attrition"),
        1.0,
    ),
    **dict.fromkeys(
        ("0", "0.0", "no", "n", "false", "f", "alive", "active", "retained", "stayed", "censored",
         "survived", "ongoing", "current", "none"),
        0.0,
    ),
}


def detect_wide_temporal_columns(columns: list[str]) -> list[str]:
    """Detect if a set of columns represents sequential time periods in headers."""
    temporal_cols: list[str] = []
    for col in columns:
        c_str = str(col).strip()
        c_clean = c_str.lower()
        if _YEAR_RE.match(c_clean) or _QUARTER_RE.match(c_clean) or c_clean in _MONTH_NAMES:
            temporal_cols.append(col)
        elif re.match(r"^\d{4}[-_/]\d{1,2}(?:[-_/]\d{1,2})?$", c_str):
            temporal_cols.append(col)

    # Need at least 4 sequential temporal columns to qualify as a wide-time table
    if len(temporal_cols) >= 4:
        return temporal_cols
    return []


def unpivot_temporal_frame(
    df: pd.DataFrame,
    temporal_columns: list[str] | None = None,
    var_name: str = "period",
    value_name: str = "value",
) -> pd.DataFrame:
    """Unpivot / melt wide temporal columns into tidy longitudinal format."""
    cols = temporal_columns or detect_wide_temporal_columns(list(df.columns))
    if not cols:
        return df.copy()

    id_vars = [c for c in df.columns if c not in cols]
    if not id_vars:
        # If all columns are temporal, preserve original index as an entity
        melted = df.reset_index().melt(
            id_vars=["index"], value_vars=cols, var_name=var_name, value_name=value_name
        )
    else:
        melted = df.melt(id_vars=id_vars, value_vars=cols, var_name=var_name, value_name=value_name)

    # Clean period values
    cleaned_periods: list[int | str] = []
    for p in melted[var_name]:
        p_str = str(p).strip()
        y_m = _YEAR_RE.match(p_str)
        if y_m:
            cleaned_periods.append(int(y_m.group(1)))
        else:
            cleaned_periods.append(p_str)

    melted[var_name] = cleaned_periods
    return melted


def detect_and_strip_summary_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Detect and remove trailing or inline subtotal/total rows."""
    if df.empty or len(df) <= 5:
        return df, []

    # Check text/categorical columns for summary tokens
    notes: list[str] = []
    mask_to_drop = pd.Series(False, index=df.index)

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            series_clean = df[col].astype(str).str.strip().str.lower()
            is_summary = series_clean.isin(_SUMMARY_TOKENS)
            if is_summary.any():
                n_matches = int(is_summary.sum())
                if n_matches <= max(3, int(len(df) * 0.05)):  # Only drop if it's a small minority of rows
                    mask_to_drop = mask_to_drop | is_summary
                    notes.append(f"Quarantined {n_matches} summary/total row(s) detected via '{col}'.")

    if mask_to_drop.any():
        cleaned = df[~mask_to_drop].copy().reset_index(drop=True)
        return cleaned, notes

    return df, []


def compute_log_returns_and_volatility(
    df: pd.DataFrame,
    price_cols: list[str] | None = None,
    date_col: str | None = None,
    window: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert raw continuous asset/sensor prices into stationary log returns and rolling volatility.

    Formula: r_t = ln(P_t / P_{t-1}), sigma_{window} = std(r_t) * sqrt(252).
    """
    out = df.copy()
    notes: list[str] = []

    if date_col and date_col in out.columns:
        try:
            out[date_col] = pd.to_datetime(out[date_col])
            out = out.sort_values(date_col).reset_index(drop=True)
        except Exception:
            pass

    targets = price_cols
    if not targets:
        # Detect price-like columns: close, adj_close, price, nav, or positive continuous measures
        targets = [
            c for c in out.columns
            if pd.api.types.is_numeric_dtype(out[c])
            and re.search(r"(?:close|price|adj_close|nav|quote)", str(c), re.I)
            and len(out[c].dropna()) >= 5
            and (out[c].dropna() > 0).all()
        ]

    for col in targets:
        s = out[col].astype(float).copy()
        valid = s.dropna()
        valid_pos = valid[valid > 0]
        if len(valid_pos) < 5:
            continue
        s_pos = s.copy()
        s_pos[s_pos <= 0] = np.nan
        ratio = s_pos / s_pos.shift(1)
        ratio[ratio <= 0] = np.nan
        log_ret = np.log(ratio)
        ret_col = f"{col}_log_return"
        vol_col = f"{col}_rolling_vol_{window}"
        out[ret_col] = log_ret.round(6)
        out[vol_col] = (log_ret.rolling(window=min(window, max(3, len(out) // 4))).std() * np.sqrt(252)).round(6)
        notes.append(f"Generated stationary log returns '{ret_col}' and annualized rolling volatility '{vol_col}' from '{col}'.")

    return out, notes


def aggregate_customer_rfm(
    df: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    amount_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate raw transactional checkout events into a tidy Customer RFM table."""
    if df.empty or customer_id_col not in df.columns or datetime_col not in df.columns or amount_col not in df.columns:
        return df.copy(), []

    working = df[[customer_id_col, datetime_col, amount_col]].dropna().copy()
    if working.empty:
        return df.copy(), []

    try:
        working[datetime_col] = pd.to_datetime(working[datetime_col])
    except Exception:
        return df.copy(), []

    ref_date = working[datetime_col].max()
    if pd.isna(ref_date):
        return df.copy(), []

    diff_days = (ref_date - working[datetime_col]).dt.total_seconds() / 86400.0
    working["_recency_days"] = diff_days.clip(lower=0)

    rfm = working.groupby(customer_id_col).agg(
        recency_days=("_recency_days", "min"),
        frequency=(amount_col, "count"),
        monetary_total=(amount_col, "sum"),
        monetary_avg=(amount_col, "mean"),
    ).reset_index()

    rfm["recency_days"] = rfm["recency_days"].astype(int)
    rfm["frequency"] = rfm["frequency"].astype(int)
    rfm["monetary_total"] = rfm["monetary_total"].round(2)
    rfm["monetary_avg"] = rfm["monetary_avg"].round(2)

    notes = [
        f"Aggregated {len(df):,} transaction rows into {len(rfm):,} customer RFM profiles "
        f"via '{customer_id_col}'."
    ]
    return rfm, notes


def pivot_basket_incidence_matrix(
    df: pd.DataFrame,
    basket_id_col: str,
    item_col: str,
    max_items: int = 50,
) -> tuple[pd.DataFrame, list[str]]:
    """Pivot long transactional receipt records into a boolean presence-absence matrix for association rules."""
    if df.empty or basket_id_col not in df.columns or item_col not in df.columns:
        return df.copy(), []

    sub = df[[basket_id_col, item_col]].dropna().copy()
    if sub.empty:
        return df.copy(), []

    # Filter to top N most frequent items for manageable matrix width
    top_items = sub[item_col].value_counts().head(max_items).index
    if len(top_items) == 0:
        return df.copy(), []

    filtered = sub[sub[item_col].isin(top_items)]
    crosstab = pd.crosstab(filtered[basket_id_col], filtered[item_col])
    incidence = (crosstab > 0).astype(int).reset_index()

    notes = [
        f"Pivoted {len(sub):,} transactions across {len(incidence):,} baskets into {len(top_items)} "
        f"boolean item-incidence indicators."
    ]
    return incidence, notes


def extract_text_sentiment_and_features(
    df: pd.DataFrame,
    text_col: str,
    max_features: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract lexical sentiment polarity, word counts, and prominent keyword features from free text."""
    if df.empty or text_col not in df.columns:
        return df.copy(), []

    out = df.copy()
    series = out[text_col].fillna("").astype(str)

    word_lists = series.str.lower().str.findall(r"\b[a-z]{3,}\b")
    out[f"{text_col}_word_count"] = word_lists.str.len()
    out[f"{text_col}_char_len"] = series.str.len()

    def _calc_polarity(words: list[str]) -> float:
        w_set = set(words)
        n_pos = len(w_set & _POSITIVE_WORDS)
        n_neg = len(w_set & _NEGATIVE_WORDS)
        denom = n_pos + n_neg
        if denom == 0:
            return 0.0
        return round((n_pos - n_neg) / denom, 3)

    out[f"{text_col}_sentiment_score"] = word_lists.apply(_calc_polarity)

    # Extract top keywords as binary occurrence features, excluding sentiment words and stop words
    all_words = [
        w for w_list in word_lists
        for w in w_list
        if w not in _POSITIVE_WORDS and w not in _NEGATIVE_WORDS and w not in _STOPWORDS and len(w) > 2
    ]
    if all_words:
        top_kws = pd.Series(all_words).value_counts().head(max_features).index
        for kw in top_kws:
            out[f"{text_col}_has_{kw}"] = series.str.lower().str.contains(rf"\b{re.escape(kw)}\b", regex=True).astype(int)

    notes = [
        f"Extracted lexical sentiment polarity, length measures, and {max_features} keyword signals from '{text_col}'."
    ]
    return out, notes


def verify_and_prepare_survival_data(
    df: pd.DataFrame,
    duration_col: str | None = None,
    event_col: str | None = None,
    event_mapping: dict[Any, int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Sanitize right-censored duration and event columns for survival modeling."""
    if df.empty:
        return df.copy(), []
    out = df.copy()
    notes: list[str] = []

    # Auto-detect duration and event columns if not provided
    if not duration_col:
        for c in out.columns:
            c_l = str(c).lower()
            if pd.api.types.is_numeric_dtype(out[c]) and any(k in c_l for k in ("tenure", "duration", "time", "months_active", "days", "follow_up")):
                duration_col = str(c)
                break

    if not event_col:
        for c in out.columns:
            c_l = str(c).lower()
            if c != duration_col and any(k in c_l for k in ("churn", "status", "event", "died", "dead", "failed", "cancelled", "censored")):
                event_col = str(c)
                break

    if not duration_col or not event_col or duration_col not in out.columns or event_col not in out.columns:
        return out, notes

    # 1. Duration validation: durations must be non-negative (>= 0).
    # Valid time-zero events (duration == 0) are preserved; negative durations (< 0) are excluded.
    dur_s = pd.to_numeric(out[duration_col], errors="coerce")
    valid_dur = (dur_s >= 0) & np.isfinite(dur_s)

    # 2. Event validation: map to 1 (event) / 0 (censored), leaving unknown / missing as NaN.
    ev_s = out[event_col]
    ev_clean: pd.Series
    if event_mapping is not None:
        ev_clean = ev_s.map(event_mapping).astype(float)
    elif pd.api.types.is_bool_dtype(ev_s):
        ev_clean = ev_s.astype(float)
    elif pd.api.types.is_numeric_dtype(ev_s):
        vals = set(pd.to_numeric(ev_s, errors="coerce").dropna().unique())
        if vals <= {0.0, 1.0}:
            ev_clean = pd.to_numeric(ev_s, errors="coerce").astype(float)
        else:
            # Non-binary numeric codes (e.g. {1, 2}) require explicit mapping
            ev_clean = pd.Series(np.nan, index=out.index, dtype=float)
            notes.append(
                f"Numeric event column '{event_col}' contains non-binary codes {sorted(vals)}; "
                "requires explicit event_mapping (e.g. {{1: 0, 2: 1}})."
            )
    else:
        text_s = ev_s.astype(str).str.strip().str.lower()
        ev_clean = text_s.map(_SURVIVAL_WORD_MAP).astype(float)
        # If the column name explicitly records censoring rather than event, invert
        c_name = str(event_col).lower()
        if "censor" in c_name:
            ev_clean = 1.0 - ev_clean

    valid_mask = valid_dur & ev_clean.notna()
    n_examined = len(df)
    n_risk = int(valid_mask.sum())
    n_excluded = n_examined - n_risk

    if n_excluded > 0:
        notes.append(
            f"Survival data prepared: {n_risk} valid records in risk set "
            f"({n_excluded} excluded due to negative duration or unknown/missing event status)."
        )
    else:
        notes.append(f"Survival data prepared: all {n_risk} records valid in risk set.")

    out = out[valid_mask].copy()
    out[duration_col] = dur_s[valid_mask]
    out[event_col] = ev_clean[valid_mask].astype(int)

    censored_share = 1.0 - float(out[event_col].mean()) if n_risk > 0 else 0.0
    notes.append(f"Survival event '{event_col}' formatted as binary (censoring rate: {censored_share:.1%}, risk set n={n_risk}).")

    return out, notes


def assess_survey_reliability_and_polarity(
    df: pd.DataFrame,
    scale_cols: list[str] | None = None,
    declared_endpoints: tuple[float, float] | None = None,
    scoring_key: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Align ordinal scale polarity and evaluate Cronbach's alpha internal consistency."""
    if df.empty:
        return df.copy(), []
    out = df.copy()
    notes: list[str] = []

    if not scale_cols:
        scale_cols = [
            c for c in out.columns
            if pd.api.types.is_numeric_dtype(out[c])
            and out[c].dropna().nunique() <= 11
            and out[c].dropna().min() >= 0
            and out[c].dropna().max() <= 10
            and len(out[c].dropna()) >= 10
        ]

    if len(scale_cols) < 2:
        return out, notes

    sub = out[scale_cols].dropna()
    if len(sub) < 5:
        return out, notes

    # Determine scale endpoints (low_end, high_end)
    low_end: float
    high_end: float
    if declared_endpoints is not None:
        low_end, high_end = float(declared_endpoints[0]), float(declared_endpoints[1])
    else:
        all_min = float(sub.min().min())
        all_max = float(sub.max().max())
        if all_min >= 1.0 and all_max <= 5.0:
            low_end, high_end = 1.0, 5.0
        elif all_min >= 1.0 and all_max <= 7.0:
            low_end, high_end = 1.0, 7.0
        elif all_min >= 0.0 and all_max <= 10.0:
            low_end, high_end = 0.0, 10.0
        else:
            low_end, high_end = all_min, all_max

    # Reverse-code negatively worded questions
    neg_phrases = (
        "difficult", "slow", "dissatisfied", "bad", "confusing", "poor",
        "complaint", "problem", "frustrat", "dislike", "hard", "negative",
    )
    aligned_cols: list[str] = []
    for c in scale_cols:
        is_negative = False
        if scoring_key is not None and c in scoring_key:
            is_negative = bool(scoring_key[c] < 0)
        else:
            is_negative = any(p in str(c).lower() for p in neg_phrases)

        if is_negative:
            rev_col = f"{c}_rev"
            out[rev_col] = (high_end + low_end) - out[c]
            notes.append(
                f"Reverse-coded negative question '{c}' -> '{rev_col}' "
                f"({low_end:.0f}..{high_end:.0f} scale)."
            )
            aligned_cols.append(rev_col)
        else:
            aligned_cols.append(c)

    # Calculate Cronbach's alpha on aligned items
    aligned_sub = out[aligned_cols].dropna()
    n_complete = len(aligned_sub)
    k = len(aligned_cols)
    if n_complete >= 5 and k >= 2:
        item_vars = float(aligned_sub.var(axis=0, ddof=1).sum())
        total_var = float(aligned_sub.sum(axis=1).var(ddof=1))
        if total_var > 0:
            alpha = (k / (k - 1)) * (1.0 - (item_vars / total_var))
            alpha = round(float(alpha), 3)
            notes.append(
                f"Computed Cronbach's alpha reliability across {k} aligned survey items "
                f"(n={n_complete} complete responses): alpha = {alpha:.2f}."
            )

    return out, notes


def compute_panel_deltas_and_lags(
    df: pd.DataFrame,
    entity_col: str | None = None,
    time_col: str | None = None,
    measure_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute within-entity deltas and lag features for longitudinal panel data."""
    if df.empty:
        return df.copy(), []
    out = df.copy()
    notes: list[str] = []

    if not entity_col:
        for c in out.columns:
            c_l = str(c).lower()
            if any(k in c_l for k in ("store", "patient", "entity", "account", "user", "id")) and out[c].nunique() < len(out):
                entity_col = str(c)
                break

    if not time_col:
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]) or re.search(r"(?:date|time|week|month|year|period)", str(c), re.I):
                time_col = str(c)
                break

    if not entity_col or not time_col or entity_col not in out.columns or time_col not in out.columns:
        return out, notes

    try:
        out = out.sort_values([entity_col, time_col]).reset_index(drop=True)
    except Exception:
        return out, notes

    if not measure_cols:
        measure_cols = [
            c for c in out.columns
            if pd.api.types.is_numeric_dtype(out[c])
            and c not in (entity_col, time_col)
            and out[c].nunique() > 5
        ][:3]

    for m in measure_cols:
        lag_col = f"{m}_lag1"
        delta_col = f"{m}_delta1"
        out[lag_col] = out.groupby(entity_col)[m].shift(1)
        out[delta_col] = out[m] - out[lag_col]
        notes.append(f"Generated within-entity lag '{lag_col}' and delta '{delta_col}' for panel entity '{entity_col}'.")

    return out, notes


def compute_geospatial_features(
    df: pd.DataFrame,
    lat_col: str | None = None,
    lon_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute distance from geographic centroid (km) with spherical dateline wrapping."""
    if df.empty:
        return df.copy(), []
    out = df.copy()
    notes: list[str] = []

    if not lat_col or not lon_col:
        for c in out.columns:
            c_l = str(c).lower()
            if not lat_col and ("lat" in c_l or c_l in ("latitude", "y")):
                lat_col = str(c)
            elif not lon_col and ("lon" in c_l or "lng" in c_l or c_l in ("longitude", "x")):
                lon_col = str(c)

    if not lat_col or not lon_col or lat_col not in out.columns or lon_col not in out.columns:
        return out, notes

    sub_lat = pd.to_numeric(out[lat_col], errors="coerce")
    sub_lon = pd.to_numeric(out[lon_col], errors="coerce")

    # Validate coordinate bounds
    valid_coords = (
        (sub_lat.abs() <= 90.0)
        & (sub_lon.abs() <= 180.0)
        & sub_lat.notna()
        & sub_lon.notna()
    )
    n_invalid = int((~valid_coords).sum())
    if n_invalid > 0:
        notes.append(f"Identified {n_invalid} invalid coordinate rows outside bounds [-90, 90] lat, [-180, 180] lon.")

    if not valid_coords.any():
        out["dist_from_centroid_km"] = np.nan
        notes.append("No valid coordinates found for geospatial distance calculation.")
        return out, notes

    center = spherical_center(
        sub_lat[valid_coords].to_numpy(dtype=float),
        sub_lon[valid_coords].to_numpy(dtype=float),
    )
    if center is None:
        out["dist_from_centroid_km"] = np.nan
        notes.append("Geographic centroid is undefined (antipodal coordinates or cancelling resultant); distance omitted.")
        return out, notes

    c_lat, c_lon = center
    dists = haversine_distance(
        sub_lat.to_numpy(dtype=float),
        sub_lon.to_numpy(dtype=float),
        c_lat,
        c_lon,
    )
    out["dist_from_centroid_km"] = np.round(dists, 3)
    notes.append(f"Computed spherical distance from centroid ({c_lat:.3f}, {c_lon:.3f}) as 'dist_from_centroid_km'.")

    return out, notes


def extract_temporal_calendar_features(
    df: pd.DataFrame,
    date_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract standard calendar and diurnal features when a time axis is present."""
    if df.empty:
        return df.copy(), []
    out = df.copy()
    notes: list[str] = []

    if not date_col:
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                date_col = str(c)
                break
        if not date_col:
            for c in out.columns:
                if any(k in str(c).lower() for k in ("date", "timestamp", "datetime")):
                    try:
                        converted = pd.to_datetime(out[c], errors="coerce")
                        if converted.notna().mean() >= 0.8:
                            out[c] = converted
                            date_col = str(c)
                            break
                    except Exception:
                        pass
    if not date_col or date_col not in out.columns:
        return out, notes

    dt_series = out[date_col]
    if not pd.api.types.is_datetime64_any_dtype(dt_series):
        dt_series = pd.to_datetime(dt_series, errors="coerce")

    if dt_series.notna().sum() < 10:
        return out, notes

    derived_any = False
    try:
        hours = dt_series.dt.hour
        if hours.nunique() > 1 and "hour_of_day" not in out.columns:
            out["hour_of_day"] = hours
            derived_any = True
            bins = [-1, 5, 11, 16, 21, 24]
            labels = ["night", "morning", "afternoon", "evening", "night_late"]
            day_parts = pd.cut(hours, bins=bins, labels=labels, ordered=False).astype(str).replace({"night_late": "night"})
            out["day_part"] = day_parts
    except Exception:
        pass

    try:
        dow = dt_series.dt.day_name()
        if dow.nunique() > 1 and "day_of_week" not in out.columns:
            out["day_of_week"] = dow
            out["is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            derived_any = True
    except Exception:
        pass

    try:
        months = dt_series.dt.month_name()
        if months.nunique() > 1 and "month_name" not in out.columns and months.nunique() <= 12:
            out["month_name"] = months
            derived_any = True
    except Exception:
        pass

    if derived_any:
        notes.append(f"Derived temporal dimensions (hour, day of week, day part) from '{date_col}'.")
    return out, notes


def discretize_continuous_quantiles(
    df: pd.DataFrame,
    metadata: Any = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create quartile bins for the primary continuous measure when no categorical dimensions exist."""
    if df.empty:
        return df.copy(), []
    out = df.copy()
    notes: list[str] = []

    has_cats = any(
        (pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]) or isinstance(out[c].dtype, pd.CategoricalDtype))
        and 2 <= out[c].nunique() <= 50
        for c in out.columns
    )
    if has_cats:
        return out, notes

    num_cols = [
        c for c in out.columns
        if pd.api.types.is_numeric_dtype(out[c])
        and not any(k in str(c).lower() for k in ("id", "index"))
        and out[c].nunique() >= 20
    ]
    if not num_cols:
        return out, notes

    target = num_cols[0]
    try:
        quartiles = pd.qcut(out[target].dropna(), q=4, labels=["Q1_Low", "Q2_MedLow", "Q3_MedHigh", "Q4_High"], duplicates="drop")
        out[f"{target}_quartile"] = quartiles
        notes.append(f"Discretized '{target}' into 4 quartiles ('{target}_quartile') to enable comparative segmentation.")
    except Exception:
        pass
    return out, notes


def aggregate_point_in_time_event_features(
    df: pd.DataFrame,
    cust_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute point-in-time cumulative event counts per entity to prevent lookahead leakage."""
    if df.empty or cust_col not in df.columns or date_col not in df.columns:
        return df.copy(), []
    out = df.copy()
    notes: list[str] = []

    timestamps = pd.to_datetime(out[date_col], errors="coerce", format="mixed", utc=True)
    # The first rank of a tied timestamp counts only strictly earlier events.
    # Missing times and unknown customers have no observable event history.
    known_customers = out[cust_col].where(timestamps.notna())
    ranks = timestamps.groupby(known_customers, sort=False).rank(method="min")
    out["cust_prior_event_count"] = (ranks - 1).astype("Int64")

    # Retain retrospective lifetime frequency for descriptive summary tables
    lifetime_freq = out[cust_col].map(out[cust_col].value_counts())
    out["cust_lifetime_frequency"] = lifetime_freq
    out["cust_event_frequency"] = lifetime_freq  # backward-compatible alias
    notes.append(
        f"Computed point-in-time prior count 'cust_prior_event_count' via '{cust_col}' "
        f"preventing lookahead leakage (lifetime frequency preserved as 'cust_lifetime_frequency')."
    )
    return out, notes


def auto_reshape_for_archetype(
    df: pd.DataFrame,
    archetype: str | DatasetArchetype,
    metadata: Any = None,
    aggregate_granularity: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Deterministic Stage 1 pipeline: cleans wide/summary layouts and applies canonical pre-modeling transforms."""
    if df.empty:
        return df, []

    # 1. Unpivot wide temporal tables if detected
    temporal_cols = detect_wide_temporal_columns(list(df.columns))
    cleaned_df = df
    notes: list[str] = []
    if temporal_cols:
        cleaned_df = unpivot_temporal_frame(cleaned_df, temporal_cols)
        notes.append(f"Unpivoted {len(temporal_cols)} wide temporal columns into tidy longitudinal format.")

    # 2. Strip summary rows if detected
    cleaned_df, strip_notes = detect_and_strip_summary_rows(cleaned_df)
    notes.extend(strip_notes)

    arch_str = str(archetype).lower()
    # 3. Canonical transformations tailored to archetype
    if arch_str in ("continuous_series", "sensor_timeseries"):
        cleaned_df, price_notes = compute_log_returns_and_volatility(cleaned_df)
        notes.extend(price_notes)
        cleaned_df, time_notes = extract_temporal_calendar_features(cleaned_df)
        notes.extend(time_notes)

    elif arch_str in ("transactional", "event_log"):
        cust_col = next((c for c in cleaned_df.columns if re.search(r"(?:customer|client|user|account)", str(c), re.I)), None)
        date_col = next((c for c in cleaned_df.columns if pd.api.types.is_datetime64_any_dtype(cleaned_df[c]) or re.search(r"(?:date|time|timestamp)", str(c), re.I)), None)
        amt_col = next((c for c in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[c]) and re.search(r"(?:amount|price|revenue|sales|spend|total)", str(c), re.I)), None)
        if cust_col and date_col and amt_col and aggregate_granularity:
            cleaned_df, rfm_notes = aggregate_customer_rfm(cleaned_df, cust_col, date_col, amt_col)
            notes.extend(rfm_notes)
        elif cust_col and date_col:
            cleaned_df, pit_notes = aggregate_point_in_time_event_features(cleaned_df, cust_col, date_col)
            notes.extend(pit_notes)
        elif cust_col:
            cust_freq = cleaned_df[cust_col].map(cleaned_df[cust_col].value_counts())
            cleaned_df["cust_lifetime_frequency"] = cust_freq
            cleaned_df["cust_event_frequency"] = cust_freq
            notes.append(f"Enriched event stream with retrospective customer frequency via '{cust_col}'.")

    elif arch_str in ("basket",):
        bsk_col = next((c for c in cleaned_df.columns if re.search(r"(?:basket|order|invoice|receipt|cart)", str(c), re.I)), None)
        itm_col = next((c for c in cleaned_df.columns if c != bsk_col and re.search(r"(?:item|product|sku|description)", str(c), re.I)), None)
        if bsk_col and itm_col:
            if aggregate_granularity:
                cleaned_df, bsk_notes = pivot_basket_incidence_matrix(cleaned_df, bsk_col, itm_col)
                notes.extend(bsk_notes)
            else:
                bsk_size = cleaned_df[bsk_col].map(cleaned_df[bsk_col].value_counts())
                cleaned_df["basket_item_count"] = bsk_size
                notes.append(f"Enriched basket items with basket size via '{bsk_col}'.")

    elif arch_str in ("survival",):
        cleaned_df, surv_notes = verify_and_prepare_survival_data(cleaned_df)
        notes.extend(surv_notes)

    elif arch_str in ("survey",):
        cleaned_df, surv_notes = assess_survey_reliability_and_polarity(cleaned_df)
        notes.extend(surv_notes)

    elif arch_str in ("panel",):
        cleaned_df, pnl_notes = compute_panel_deltas_and_lags(cleaned_df)
        notes.extend(pnl_notes)

    elif arch_str in ("geospatial",):
        cleaned_df, geo_notes = compute_geospatial_features(cleaned_df)
        notes.extend(geo_notes)

    elif arch_str in ("hybrid_text",):
        text_cols = [
            c for c in cleaned_df.columns
            if (pd.api.types.is_object_dtype(cleaned_df[c]) or pd.api.types.is_string_dtype(cleaned_df[c]))
            and cleaned_df[c].astype(str).str.len().mean() >= 40.0
            and cleaned_df[c].nunique() > 5
        ]
        if text_cols:
            cleaned_df, text_notes = extract_text_sentiment_and_features(cleaned_df, text_cols[0])
            notes.extend(text_notes)

    # For any archetype with datetime columns and sparse categorical features, enrich with calendar & quantile features
    cat_cols = [
        c for c in cleaned_df.columns
        if pd.api.types.is_object_dtype(cleaned_df[c])
        or pd.api.types.is_string_dtype(cleaned_df[c])
        or isinstance(cleaned_df[c].dtype, pd.CategoricalDtype)
    ]
    if len(cat_cols) <= 1:
        cleaned_df, time_notes = extract_temporal_calendar_features(cleaned_df)
        notes.extend(time_notes)
        cleaned_df, q_notes = discretize_continuous_quantiles(cleaned_df)
        notes.extend(q_notes)

    return cleaned_df, notes

