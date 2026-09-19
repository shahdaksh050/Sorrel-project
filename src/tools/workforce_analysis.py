"""
Workforce / HR Analysis Tool — Execution Layer.

Headcount, tenure, attrition and pay-distribution measures for datasets
the domain layer identified as workforce data (src/core/domains.py).

Why a dedicated tool: HR tables are one row per person, so the useful
questions are about distribution and composition rather than correlation
— what tenure looks like across the organisation, where attrition
concentrates, how pay is spread within and between groups. The generic
numeric summary reports a mean salary and stops, which is both the least
robust summary of a skewed distribution and the least actionable.

Pay comparisons here are deliberately **unadjusted**: they compare raw
medians between groups without controlling for role, level or tenure. An
unadjusted gap is a real and reportable fact about what people are paid,
but it is not evidence of unequal pay for equal work, and the output says
so rather than letting a reader infer the stronger claim.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.core.domains import domain_confidence, resolve_column
from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: Groups smaller than this are reported but excluded from pay-gap
#: comparisons — a median over a handful of people is noise, and naming it
#: invites a conclusion the data cannot support.
_MIN_GROUP_FOR_COMPARISON = 5

#: Token sets that mark a status value as "no longer employed".
_INACTIVE_TOKENS = frozenset(
    {"left", "leaver", "terminated", "resigned", "exited", "inactive",
     "separated", "quit", "churned", "no", "false", "0"}
)
_ACTIVE_TOKENS = frozenset({"active", "employed", "current", "yes", "true", "1"})
#: Affirmative tokens. Their meaning depends on the column: "Attrition=Yes"
#: means the person left, "Active=Yes" means they stayed.
_AFFIRMATIVE_TOKENS = frozenset({"yes", "true", "1", "y"})
#: Column-name fragments marking a departure flag (affirmative = departed).
_DEPARTURE_NAME_TOKENS = (
    "attrition", "left", "churn", "terminated", "exited", "quit",
    "separated", "resigned",
)

_TOP_N = 12


def _describe_pay(series: pd.Series) -> dict[str, float]:
    """Distribution summary that survives the skew typical of salary data."""
    clean = series.dropna()
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    return {
        "count": int(clean.size),
        "median": round(float(clean.median()), 2),
        "mean": round(float(clean.mean()), 2),
        "p10": round(float(clean.quantile(0.10)), 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "p90": round(float(clean.quantile(0.90)), 2),
        "min": round(float(clean.min()), 2),
        "max": round(float(clean.max()), 2),
        # Ratio of the top decile to the bottom decile — a compact, outlier-
        # resistant read on internal pay dispersion.
        "p90_p10_ratio": round(
            float(clean.quantile(0.90)) / float(clean.quantile(0.10)), 3
        )
        if float(clean.quantile(0.10)) > 0
        else 0.0,
    }


def _pay_gap_between(
    work: pd.DataFrame, salary_column: str, grouping_column: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Unadjusted median-pay comparison across the levels of
    `grouping_column` (department, gender, or any other categorical split).

    Returns `(pay_by_level, gap)`: `pay_by_level` is every level's
    count/median/mean pay, sorted by median descending; `gap` compares the
    highest- and lowest-paid levels with enough headcount to be a real
    comparison (`_MIN_GROUP_FOR_COMPARISON`), or is `None` when fewer than
    two levels qualify.
    """
    levels: list[dict[str, Any]] = []
    for name, group in work.groupby(grouping_column)[salary_column]:
        if group.notna().sum() == 0:
            continue
        levels.append(
            {
                "group": str(name),
                "count": int(group.notna().sum()),
                "median_pay": round(float(group.median()), 2),
                "mean_pay": round(float(group.mean()), 2),
            }
        )
    levels.sort(key=lambda g: g["median_pay"], reverse=True)

    comparable = [g for g in levels if g["count"] >= _MIN_GROUP_FOR_COMPARISON]
    gap: dict[str, Any] | None = None
    if len(comparable) >= 2:
        high, low = comparable[0], comparable[-1]
        if high["median_pay"] > 0:
            gap_frac = (high["median_pay"] - low["median_pay"]) / high["median_pay"]
            gap = {
                "higher_group": high["group"],
                "lower_group": low["group"],
                "gap_pct": round(gap_frac * 100, 2),
                "higher_median": high["median_pay"],
                "lower_median": low["median_pay"],
                "note": (
                    "Unadjusted: not controlled for role, level or tenure. "
                    f"Groups smaller than {_MIN_GROUP_FOR_COMPARISON} were "
                    "excluded from this comparison."
                ),
            }
    return levels, gap


def _is_departure_named(name: object) -> bool:
    """True when the column name says an affirmative value means departed."""
    parts = re.split(r"[^a-z]+", str(name).lower())
    return any(p in _DEPARTURE_NAME_TOKENS for p in parts) or any(
        t in str(name).lower() for t in ("attrition", "churn")
    )


def _resolve_status(series: pd.Series) -> pd.Series | None:
    """Map a status/attrition column to True=departed, False=active.

    Polarity comes from the column name: on a departure-named column
    (Attrition, Left, Churn, ...) yes/true/1 means departed; on a
    status/active-named column yes/true/1 means still employed.

    Returned as a nullable ``boolean`` series: a row whose status is missing
    stays ``<NA>`` (unknown) rather than being read as active, so it drops
    out of every attrition numerator and denominator.
    """
    departure = _is_departure_named(series.name)
    missing = series.isna()
    if pd.api.types.is_bool_dtype(series):
        flags = series.astype("boolean")
        return flags if departure else ~flags
    tokens = series.dropna().astype(str).str.strip().str.lower()
    if tokens.empty:
        return None
    distinct = set(tokens.unique())
    if distinct & _INACTIVE_TOKENS or distinct & _ACTIVE_TOKENS:
        normalised = series.astype(str).str.strip().str.lower()
        if departure:
            inactive = (_INACTIVE_TOKENS - {"no", "false", "0"}) | _AFFIRMATIVE_TOKENS
            return normalised.isin(inactive).astype("boolean").mask(missing)
        return normalised.isin(_INACTIVE_TOKENS).astype("boolean").mask(missing)
    return None


class WorkforceAnalysisTool(BaseTool):
    """Headcount, tenure, attrition and pay-distribution analysis for HR data."""

    name = "workforce_analysis"
    description = (
        "Analyse employee/HR records: headcount and composition by department "
        "and job title, tenure distribution, attrition rate and where it "
        "concentrates, and pay distribution including unadjusted pay gaps "
        "between groups. Use for people/roster data with one row per employee."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return domain_confidence(profile, "workforce")

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        salary_column: str | None = None,
        department_column: str | None = None,
        hire_date_column: str | None = None,
        exit_date_column: str | None = None,
        status_column: str | None = None,
        group_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        # Sequential resolution with each claimed column withdrawn from the
        # pool — hire/exit dates and status flags share name fragments, and a
        # column filling two roles would silently corrupt tenure or attrition.
        claimed: set[str] = {
            c for c in (salary_column, department_column, hire_date_column,
                        exit_date_column, status_column, group_column) if c
        }
        salary_column = salary_column or resolve_column(
            df, ("salary", "compensation", "pay", "wage", "income", "ctc"), claimed)
        claimed.add(salary_column or "")
        department_column = department_column or resolve_column(
            df, ("department", "dept", "division", "team", "function"), claimed)
        claimed.add(department_column or "")
        hire_date_column = hire_date_column or resolve_column(
            df, ("hire_date", "joining_date", "start_date", "joined", "doj", "hire"), claimed)
        claimed.add(hire_date_column or "")
        exit_date_column = exit_date_column or resolve_column(
            df, ("exit_date", "termination_date", "leave_date", "resignation_date",
                 "end_date"), claimed)
        claimed.add(exit_date_column or "")
        status_column = status_column or resolve_column(
            df, ("employment_status", "status", "attrition", "active", "left",
                 "churn"), claimed)
        claimed.add(status_column or "")
        group_column = group_column or resolve_column(df, ("gender", "sex"), claimed)

        if salary_column is None or salary_column not in df.columns:
            raise ToolExecutionError(
                "No compensation column found. Pass salary_column explicitly."
            )

        work = df.copy()
        work[salary_column] = pd.to_numeric(work[salary_column], errors="coerce")
        if work[salary_column].notna().sum() == 0:
            raise ToolExecutionError(
                f"Column '{salary_column}' has no numeric values to analyse."
            )

        headcount = len(work)
        result: dict[str, Any] = {
            "headcount": headcount,
            "salary_column": salary_column,
            "pay_distribution": _describe_pay(work[salary_column]),
            "pay_comparison_caveat": (
                "Pay figures are unadjusted: they compare raw medians without "
                "controlling for role, level, tenure or location. A gap here "
                "describes what groups are paid, not whether equal work is paid "
                "equally."
            ),
        }

        # ---- Attrition ----
        departed: pd.Series | None = None
        if status_column and status_column in work.columns:
            departed = _resolve_status(work[status_column])
            if departed is not None:
                result["status_column"] = status_column
        if departed is None and exit_date_column and exit_date_column in work.columns:
            parsed_exit = pd.to_datetime(work[exit_date_column], errors="coerce")
            departed = parsed_exit.notna()
            result["exit_date_column"] = exit_date_column

        if departed is not None:
            # Rows with a missing status are unknown, not active: they are
            # excluded from both the leaver count and the denominator.
            status_known = int(departed.notna().sum())
            if status_known == 0:
                departed = None
        if departed is not None:
            leavers = int(departed.sum())
            result["departed_count"] = leavers
            result["active_count"] = status_known - leavers
            result["status_known_count"] = status_known
            result["attrition_rate_pct"] = round(leavers / status_known * 100, 2)
            result["attrition_basis"] = (
                "Share of records with a known status that are marked as departed "
                f"({status_known:,} of {headcount:,} records) — a cumulative rate over "
                "the whole file, not an annualised turnover rate."
            )

        # ---- Tenure ----
        if hire_date_column and hire_date_column in work.columns:
            hired = pd.to_datetime(work[hire_date_column], errors="coerce")
            if hired.notna().any():
                # Measure to each person's exit where known, otherwise to the
                # latest date present in the file — never to today's date, so
                # the number is reproducible as the file ages.
                end = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
                if exit_date_column and exit_date_column in work.columns:
                    end = pd.to_datetime(work[exit_date_column], errors="coerce")
                reference = max(
                    [d for d in (hired.max(), end.max()) if pd.notna(d)],
                    default=pd.Timestamp.now(),
                )
                # A known leaver with a missing exit date has an unknown end,
                # not one at the reference date — leave it NaT (excluded).
                unknown_end = pd.Series(False, index=work.index)
                if departed is not None and exit_date_column and exit_date_column in work.columns:
                    unknown_end = end.isna() & departed.fillna(False).astype(bool)
                end = end.fillna(reference).mask(unknown_end)
                tenure_years = (end - hired).dt.total_seconds() / (365.25 * 86400)
                tenure_years = tenure_years[tenure_years.notna() & (tenure_years >= 0)]
                if not tenure_years.empty:
                    result["hire_date_column"] = hire_date_column
                    result["tenure_years"] = {
                        "count": int(tenure_years.size),
                        "median": round(float(tenure_years.median()), 2),
                        "mean": round(float(tenure_years.mean()), 2),
                        "p10": round(float(tenure_years.quantile(0.10)), 2),
                        "p90": round(float(tenure_years.quantile(0.90)), 2),
                        "max": round(float(tenure_years.max()), 2),
                    }
                    result["tenure_reference_date"] = str(reference.date())
                    result["share_under_1_year_pct"] = round(
                        float((tenure_years < 1).mean()) * 100, 2
                    )
                    result["share_over_5_years_pct"] = round(
                        float((tenure_years > 5).mean()) * 100, 2
                    )

        # ---- Composition and pay by department ----
        if department_column and department_column in work.columns:
            result["department_column"] = department_column
            by_dept = work.groupby(department_column)[salary_column]
            rows: list[dict[str, Any]] = []
            # Per-department count of rows with a known status — the
            # attrition denominator, used for the comparison-size floor.
            status_known_by_dept: dict[str, int] = {}
            for name, group in by_dept:
                entry: dict[str, Any] = {
                    "department": str(name),
                    "headcount": int(group.size),
                    "headcount_share_pct": round(int(group.size) / headcount * 100, 2),
                }
                if group.notna().any():
                    entry["median_pay"] = round(float(group.median()), 2)
                if departed is not None:
                    mask = work[department_column] == name
                    dept_status = departed[mask].dropna()
                    status_known_by_dept[str(name)] = int(dept_status.size)
                    if not dept_status.empty:
                        entry["attrition_rate_pct"] = round(
                            float(dept_status.astype(float).mean()) * 100, 2
                        )
                rows.append(entry)
            rows.sort(key=lambda r: r["headcount"], reverse=True)
            result["by_department"] = rows[:_TOP_N]
            if departed is not None:
                eligible = [
                    r for r in rows
                    if "attrition_rate_pct" in r
                    and status_known_by_dept.get(r["department"], 0) >= _MIN_GROUP_FOR_COMPARISON
                ]
                if eligible:
                    worst = max(eligible, key=lambda r: r["attrition_rate_pct"])
                    result["highest_attrition_department"] = {
                        "department": worst["department"],
                        "attrition_rate_pct": worst["attrition_rate_pct"],
                        "headcount": worst["headcount"],
                    }

            # Unadjusted pay gap between the highest- and lowest-paid
            # departments — department is itself a "group" worth comparing
            # pay across, same mechanism as the gender/sex comparison below.
            # `by_department` above already has median pay per department
            # but never compared them against each other; without this, a
            # dataset with no gender/sex column (most HR rosters don't have
            # one) never got ANY pay-gap finding at all, even when the
            # department breakdown made one obvious.
            # Not `_` — execute()'s own `**_: Any` kwargs catch-all already
            # binds that name as dict[str, Any] in this scope; reusing it
            # here for a throwaway list would conflict under mypy.
            _dept_pay_by_level, dept_gap = _pay_gap_between(work, salary_column, department_column)
            if dept_gap:
                result["department_pay_gap"] = dept_gap

        # ---- Unadjusted pay comparison between groups (gender/sex) ----
        if group_column and group_column in work.columns:
            result["group_column"] = group_column
            pay_by_group, group_gap = _pay_gap_between(work, salary_column, group_column)
            result["pay_by_group"] = pay_by_group
            if group_gap:
                result["unadjusted_median_pay_gap"] = group_gap

        pay = result["pay_distribution"]
        parts = [f"{headcount:,} employee records; median pay {pay['median']:,.2f}"]
        if "attrition_rate_pct" in result:
            parts.append(f"{result['attrition_rate_pct']:.1f}% departed")
        if "tenure_years" in result:
            parts.append(f"median tenure {result['tenure_years']['median']:.1f} years")
        if "department_pay_gap" in result:
            gap = result["department_pay_gap"]
            parts.append(
                f"unadjusted median pay gap {gap['gap_pct']:.1f}% between departments "
                f"({gap['higher_group']} vs {gap['lower_group']})"
            )
        if "unadjusted_median_pay_gap" in result:
            gap = result["unadjusted_median_pay_gap"]
            parts.append(
                f"unadjusted median pay gap {gap['gap_pct']:.1f}% "
                f"({gap['higher_group']} vs {gap['lower_group']})"
            )
        result["summary"] = "; ".join(parts) + "."
        return result

    def _pay_gap_finding(self, gap: dict[str, Any], dimension: str) -> Finding | None:
        """Build a `segment_lift`/`median_pay` Finding from a
        `_pay_gap_between()` result, or `None` below the triviality floor."""
        gap_pct = gap.get("gap_pct", 0.0)
        if gap_pct <= 2.0:
            return None
        note = gap.get("note", "")
        return Finding(
            finding_id=f"{self.name}_pay_gap_{dimension}",
            kind="segment_lift",
            headline=(
                f"Unadjusted median pay gap of {gap_pct:.1f}% between "
                f"{gap.get('higher_group')} and {gap.get('lower_group')}"
            ),
            detail=note,
            evidence=dict(gap),
            measure="median_pay",
            dimension=dimension,
            level=f"{gap.get('higher_group')} vs {gap.get('lower_group')}",
            effect=round(gap_pct / 100.0, 4),
            effect_kind="pct",
            confidence=0.55,
            surprise=0.5,
            caveats=[note] if note else [],
        )

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []

        summary = output.get("summary")
        if summary:
            results.append(
                Finding(
                    finding_id=f"{self.name}_summary",
                    kind="workforce",
                    headline=summary.rstrip("."),
                    evidence={
                        "headcount": output.get("headcount"),
                        "pay_distribution": output.get("pay_distribution"),
                    },
                    measure="headcount",
                    confidence=0.7,
                    surprise=0.2,
                    objective_fit=0.25,
                    layer="exec",
                )
            )

        # Department and gender/sex pay gaps are independent findings — a
        # dataset can have either, both, or neither, so each is built from
        # its own `output` field via the same helper rather than assuming
        # only the gender-style `group_column` gap exists.
        dept_gap = output.get("department_pay_gap")
        if dept_gap is not None:
            finding = self._pay_gap_finding(
                dept_gap, dimension=output.get("department_column") or "department"
            )
            if finding is not None:
                results.append(finding)

        group_gap = output.get("unadjusted_median_pay_gap")
        if group_gap is not None:
            finding = self._pay_gap_finding(
                group_gap, dimension=output.get("group_column") or "group"
            )
            if finding is not None:
                results.append(finding)

        highest_attrition = output.get("highest_attrition_department")
        overall_attrition = output.get("attrition_rate_pct")
        if highest_attrition:
            dept_rate = highest_attrition.get("attrition_rate_pct", 0.0)
            lift = dept_rate - overall_attrition if overall_attrition is not None else dept_rate
            if lift > 3.0:
                results.append(
                    Finding(
                        finding_id=f"{self.name}_attrition_{highest_attrition.get('department')}",
                        kind="segment_lift",
                        headline=(
                            f"'{highest_attrition.get('department')}' has the highest attrition "
                            f"at {dept_rate:.1f}%"
                            + (
                                f" vs {overall_attrition:.1f}% overall"
                                if overall_attrition is not None
                                else ""
                            )
                        ),
                        evidence={
                            **highest_attrition,
                            "overall_attrition_rate_pct": overall_attrition,
                        },
                        measure="attrition_rate",
                        dimension="department",
                        level=str(highest_attrition.get("department")),
                        effect=round(lift / 100.0, 4),
                        effect_kind="pct",
                        confidence=0.6,
                        surprise=0.5,
                    )
                )

        tenure = output.get("tenure_years")
        under1 = output.get("share_under_1_year_pct")
        if tenure and under1 is not None and under1 > 15.0:
            results.append(
                Finding(
                    finding_id=f"{self.name}_tenure",
                    kind="workforce",
                    headline=(
                        f"{under1:.1f}% of employees have under 1 year of tenure "
                        f"(median tenure {tenure.get('median')} years)"
                    ),
                    evidence={
                        "tenure_years": tenure,
                        "share_under_1_year_pct": under1,
                        "share_over_5_years_pct": output.get("share_over_5_years_pct"),
                    },
                    measure="tenure",
                    effect=round(under1 / 100.0, 4),
                    effect_kind="pct",
                    confidence=0.6,
                    surprise=0.3,
                )
            )

        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "salary_column": {
                "type": "string",
                "description": "Compensation column. Auto-detected when omitted.",
                "required": False,
            },
            "department_column": {
                "type": "string",
                "description": "Department/division column for composition breakdown.",
                "required": False,
            },
            "hire_date_column": {
                "type": "string",
                "description": "Hire/joining date, used to compute tenure.",
                "required": False,
            },
            "exit_date_column": {
                "type": "string",
                "description": (
                    "Exit/termination date. Used for attrition when no status "
                    "column exists, and to end tenure for leavers."
                ),
                "required": False,
            },
            "status_column": {
                "type": "string",
                "description": (
                    "Employment status / attrition flag (active vs left). "
                    "Auto-detected when omitted."
                ),
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": (
                    "Categorical column to compare unadjusted pay across "
                    "(e.g. gender). Defaults to a detected gender column."
                ),
                "required": False,
            },
        }

