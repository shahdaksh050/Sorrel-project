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
            if pd.api.types.is_numeric_dtype(out[c]) and any(k in c_l for k in ("tenure", "duration", "time", "months_active", "days")):
                duration_col = str(c)
                break

    if not event_col:
        for c in out.columns:
            c_l = str(c).lower()
            if c != duration_col and any(k in c_l for k in ("churn", "status", "event", "died", "failed", "cancelled")):
                event_col = str(c)
                break

    if duration_col and duration_col in out.columns:
        dur_s = pd.to_numeric(out[duration_col], errors="coerce")
        invalid_dur = int((dur_s <= 0).sum())
        if invalid_dur > 0:
            out[duration_col] = dur_s.clip(lower=0.01)
            notes.append(f"Adjusted {invalid_dur} non-positive '{duration_col}' values to 0.01 for survival model validity.")

    if event_col and event_col in out.columns:
        ev_s = out[event_col]
        if pd.api.types.is_bool_dtype(ev_s):
            out[event_col] = ev_s.astype(int)
        elif pd.api.types.is_object_dtype(ev_s) or pd.api.types.is_string_dtype(ev_s):
            lower_ev = ev_s.astype(str).str.strip().str.lower()
            pos_mask = lower_ev.isin({"1", "true", "yes", "churn", "churned", "died", "dead", "failed", "cancelled"})
            out[event_col] = pos_mask.astype(int)
        ev_num = pd.to_numeric(out[event_col], errors="coerce").fillna(0)
        censored_share = 1.0 - float(ev_num.mean()) if len(ev_num) > 0 else 0.0
        notes.append(f"Survival event '{event_col}' formatted as binary (censoring rate: {censored_share:.1%}).")

    return out, notes


def assess_survey_reliability_and_polarity(
    df: pd.DataFrame,
    scale_cols: list[str] | None = None,
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

    # Reverse-code negatively worded questions
    neg_phrases = ("difficult", "slow", "dissatisfied", "bad", "confusing", "poor", "complaint", "problem")
    for c in scale_cols:
        if any(p in str(c).lower() for p in neg_phrases):
            col_min = float(out[c].min())
            col_max = float(out[c].max())
            rev_col = f"{c}_rev"
            out[rev_col] = (col_max + col_min) - out[c]
            notes.append(f"Reverse-coded negative question '{c}' -> '{rev_col}' ({col_min:.0f}..{col_max:.0f} range).")

    # Calculate Cronbach's alpha
    k = len(scale_cols)
    item_vars = float(sub.var(axis=0, ddof=1).sum())
    total_var = float(sub.sum(axis=1).var(ddof=1))
    if total_var > 0:
        alpha = (k / (k - 1)) * (1.0 - (item_vars / total_var))
        alpha = round(float(alpha), 3)
        notes.append(f"Computed Cronbach's alpha reliability across {k} survey items: alpha = {alpha:.2f}.")

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
    """Compute distance from geographic centroid (km) and spatial bounds."""
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
    mean_lat = float(sub_lat.mean())
    mean_lon = float(sub_lon.mean())

    # Haversine distance from centroid in km
    dlat = np.radians(sub_lat - mean_lat)
    dlon = np.radians(sub_lon - mean_lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(mean_lat)) * np.cos(np.radians(sub_lat)) * np.sin(dlon / 2.0) ** 2
    c_dist = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    out["dist_from_centroid_km"] = (6371.0 * c_dist).round(3)
    notes.append(f"Computed spatial distance from centroid ({mean_lat:.3f}, {mean_lon:.3f}) as 'dist_from_centroid_km'.")

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
        if cust_col and date_col and amt_col:
            if aggregate_granularity:
                cleaned_df, rfm_notes = aggregate_customer_rfm(cleaned_df, cust_col, date_col, amt_col)
                notes.extend(rfm_notes)
            else:
                cust_freq = cleaned_df[cust_col].map(cleaned_df[cust_col].value_counts())
                cleaned_df["cust_event_frequency"] = cust_freq
                notes.append(f"Enriched event stream with customer frequency via '{cust_col}'.")

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

