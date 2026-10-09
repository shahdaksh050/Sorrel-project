"""
Multiple-comparison correction (item 4 / U0.7 follow-on).

A run can call select_statistical_test repeatedly across many
feature/group pairs; uncorrected, each test keeps its own 0.05
false-positive rate and the run-wide false-discovery rate climbs with
every additional test. This corrects across every p-value a run actually
produced (accumulated by the controller's execution loop into memory
context "statistical_test_pvalues") at report time — both
src.tools.report_generator (Markdown) and src.core.html_report (HTML) use
it so the two reports agree.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.findings import Finding

#: Default significance level — matches SelectStatisticalTestTool's default
#: alpha; a run using a different per-call alpha still gets one consistent
#: correction here.
DEFAULT_ALPHA = 0.05


def apply_benjamini_hochberg(
    pvalue_tests: list[dict[str, Any]], alpha: float = DEFAULT_ALPHA
) -> list[dict[str, Any]]:
    """Each dict must have a "p_value" key; every other key is passed
    through unchanged. Adds "p_adjusted" and "significant_after_correction".

    Non-finite/out-of-range values are retained as non-testable rows with a
    null adjusted p-value; they never contaminate the valid BH family.
    """
    if not pvalue_tests:
        return []
    from statsmodels.stats.multitest import multipletests

    valid: list[tuple[int, float]] = []
    for index, test in enumerate(pvalue_tests):
        value = test.get("p_value")
        if value is None or isinstance(value, bool):
            continue
        try:
            p_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(p_value) and 0.0 <= p_value <= 1.0:
            valid.append((index, p_value))

    result = [
        {**test, "p_adjusted": None, "significant_after_correction": False}
        for test in pvalue_tests
    ]
    valid_indexes = {index for index, _p_value in valid}
    for index, row in enumerate(result):
        if index not in valid_indexes:
            # JSON's NaN/Infinity extensions are non-standard and frequently
            # misread as evidence. Normalise an undefined test statistic to
            # an explicit null at this boundary.
            row["p_value"] = None
    if not valid:
        return result

    pvals = [p_value for _index, p_value in valid]
    reject, p_adjusted, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    for (index, _p_value), adjusted, rejected in zip(valid, p_adjusted, reject, strict=True):
        result[index]["p_adjusted"] = round(float(adjusted), 6)
        result[index]["significant_after_correction"] = bool(rejected)
    return result


def adjust_findings_run_level(findings: list[Finding], alpha: float = DEFAULT_ALPHA) -> int:
    """
    One BH family per run: every finding carrying a p-value, whichever tool
    produced it. The tool's own (within-tool) correction is kept once in
    evidence["p_adjusted_within_tool"] before the first overwrite; the
    finding's `p_adjusted` becomes the stricter of the run-level and
    within-tool values (a correction never makes a finding look more
    significant than its own tool judged it), which ranking reads.
    Returns the family size.
    """
    family = [
        f for f in findings
        if isinstance(f.p_value, (int, float)) and not isinstance(f.p_value, bool)
        and math.isfinite(f.p_value) and 0.0 <= f.p_value <= 1.0
    ]
    if not family:
        return 0
    adjusted = apply_benjamini_hochberg([{"p_value": f.p_value} for f in family], alpha=alpha)
    for f, row in zip(family, adjusted, strict=True):
        if not isinstance(f.evidence, dict):
            f.evidence = {}
        if "p_adjusted_within_tool" not in f.evidence:
            f.evidence["p_adjusted_within_tool"] = f.p_adjusted
        within = f.evidence["p_adjusted_within_tool"]
        f.p_adjusted = (
            max(row["p_adjusted"], float(within))
            if isinstance(within, (int, float)) and not isinstance(within, bool)
            else row["p_adjusted"]
        )
    return len(family)
