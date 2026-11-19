"""Operator-answer regression cases supplementing tests/test_planted_effects.py.

Covers all 10 universal archetypes across deterministic and recorded planning,
including positive controls with genuine planted signals and negative / unanswerable
controls verifying inconclusive / unperformed checks, zero fabricated metrics,
and proper boundary caveats.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController
from src.core.geo_stats import spherical_center
from src.core.reshaping import (
    aggregate_point_in_time_event_features,
    assess_survey_reliability_and_polarity,
    compute_panel_deltas_and_lags,
    verify_and_prepare_survival_data,
)
from src.tools.base import ToolExecutionError
from src.tools.basket import BasketAnalysisTool
from src.tools.cohort_analysis import CohortAnalysisTool
from src.tools.forecast import ForecastAnalysisTool
from src.tools.geospatial import GeospatialAnalysisTool
from src.tools.graph_analysis import GraphAnalysisTool
from src.tools.mixed_model import MixedModelAnalysisTool
from src.tools.segment_comparison import SegmentComparisonTool
from src.tools.survival import SurvivalAnalysisTool
from src.tools.text_analysis import TextAnalysisTool
from tests.test_controller import _ScriptedLLM

ALL_ARCHETYPE_FAMILIES: list[str] = [
    "survival",
    "basket",
    "network",
    "geospatial",
    "survey",
    "transactional",
    "continuous_series",
    "panel",
    "cross_sectional",
    "hybrid_text",
]


def _reference_case(family: str) -> tuple[pd.DataFrame, str, dict[str, str], str]:
    if family == "survival":
        return pd.DataFrame({
            "account_id": [f"A{i}" for i in range(80)],
            "tenure_months": [1] * 20 + [2] * 20 + [3] * 40,
            "churned": [1] * 40 + [0] * 40,
        }), "survival_analysis", {"duration_column": "tenure_months", "event_column": "churned"}, "How long until churn?"
    if family == "basket":
        rows = [(f"B{i}", item) for i in range(80) for item in (["tea", "milk", "tea"] if i < 20 else ["bread", "salt"])]
        return pd.DataFrame(rows, columns=["basket_id", "item"]), "basket_analysis", {
            "order_column": "basket_id", "item_column": "item",
        }, "Which items are bought together?"
    if family == "network":
        rows = [(f"N{i % 6}", "hub") for i in range(60)] + [("hub", "sink")] * 15
        return pd.DataFrame(rows, columns=["source_id", "target_id"]), "graph_analysis", {
            "source_column": "source_id", "target_column": "target_id",
        }, "Which nodes connect the network?"
    if family == "geospatial":
        return pd.DataFrame({
            "ride_id": [f"R{i}" for i in range(60)],
            "pickup_latitude": np.tile([-1.0, 1.0], 30),
            "pickup_longitude": np.tile([-179.0, 179.0], 30),
        }), "geospatial_analysis", {}, "Where are pickups concentrated?"
    if family == "survey":
        responses = np.tile([1, 2, 3, 4, 5], 12)
        return pd.DataFrame({
            "respondent_id": [f"R{i}" for i in range(60)], "q1_service": responses,
            "q2_support": responses[::-1], "q3_value": np.roll(responses, 1),
            "q4_design": np.roll(responses, 2),
        }), "clean_data", {}, "Describe the survey item responses."
    if family == "transactional":
        timestamps = [
            pd.Timestamp("2024-01-01 08:14:00") + pd.Timedelta(minutes=int(i * 17 + (i % 5) * 3))
            for i in range(80)
        ]
        return pd.DataFrame({
            "order_id": [f"ORD_{i // 2:04d}" for i in range(80)],
            "customer_id": [f"CUST_{i % 8:02d}" for i in range(80)],
            "timestamp": timestamps,
            "amount": [100.0 if i % 8 != 7 else -40.0 for i in range(80)],
            "quantity": [1] * 80,
        }), "cohort_analysis", {
            "date_column": "timestamp", "amount_column": "amount",
            "order_column": "order_id", "customer_column": "customer_id",
        }, "What is the repeat customer behavior and line item revenue?"
    if family == "continuous_series":
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        return pd.DataFrame({
            "timestamp": dates,
            "demand": [100.0 + 2.0 * i for i in range(60)],
        }), "time_series_analysis", {
            "date_column": "timestamp", "value_column": "demand",
        }, "What is the demand trend and change over time?"
    if family == "panel":
        dates = pd.date_range("2024-01-01", periods=12, freq="W")
        records: list[dict[str, Any]] = []
        for i in range(6):
            store = f"Store_{i}"
            base = 1000.0 * (i + 1)
            for d in dates:
                records.append({
                    "store_id": store,
                    "week_date": d,
                    "sales": float(base + (i * 20)),
                })
        return pd.DataFrame(records), "change_analysis", {
            "date_column": "week_date", "measure_column": "sales", "dimension_column": "store_id",
        }, "What is the change in sales across stores?"
    if family == "cross_sectional":
        return pd.DataFrame({
            "employee_id": [f"E{i}" for i in range(80)],
            "department": ["Eng"] * 40 + ["Sales"] * 40,
            "salary": [120000.0 if i < 40 else 80000.0 for i in range(80)],
        }), "segment_comparison", {
            "measure_column": "salary", "dimension_column": "department",
        }, "Compare salary across departments."
    if family == "hybrid_text":
        comments = [
            "The software architecture is well thought out, fast and highly extensible for data science.",
            "Constant crashes during large file exports, completely unreliable and lacks customer support.",
            "Decent user interface but sluggish when loading large dashboards with complex visualizations.",
            "Exceptional performance, smooth animations, and clean type-safe python implementation throughout.",
            "Terrible documentation, ambiguous error messages and very confusing navigation across views.",
        ]
        return pd.DataFrame({
            "ticket_id": [f"TCK_{i:03d}" for i in range(50)],
            "review_text": [comments[i % len(comments)] for i in range(50)],
            "score": [i % 5 + 1 for i in range(50)],
        }), "text_analysis", {
            "text_column": "review_text",
        }, "What are the key customer feedback themes in the reviews?"
    raise ValueError(f"Unknown archetype family: {family}")


@pytest.mark.parametrize("family", ALL_ARCHETYPE_FAMILIES)
@pytest.mark.parametrize("planner", ["deterministic", "recorded"])
def test_operator_report_reference_case(family: str, planner: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table, tool, parameters, objective = _reference_case(family)
    path = tmp_path / f"{family}.csv"
    table.to_csv(path, index=False)
    output = tmp_path / "output"
    monkeypatch.setenv("CHART_DESIGN", "false")
    controller = AgentController(objective=objective, output_dir=str(output), use_ml=False,
                                 use_llm=False, max_iterations=2, min_iterations=1)
    controller.load_dataset(str(path), interactive=False)
    if planner == "recorded":
        controller.use_llm = True
        controller.llm_client = _ScriptedLLM([  # type: ignore[assignment]
            {"status": "in_progress", "steps": [{"step_number": 1, "tool_name": tool,
              "parameters": parameters, "rationale": objective}]},
            {"status": "complete", "reasoning": "Unverified projection of doubled profits.",
             "insights": [], "recommendations": [], "key_metrics": {}},
        ])
    result = controller.analyze()
    reports = output / "reports"
    raw = json.loads((reports / f"{family}_raw.json").read_text(encoding="utf-8"))
    delivered = raw["llm_insights"]
    assert delivered["coverage"] == result["coverage"]
    assert delivered["executive_summary"] == result["executive_summary"]
    for filename in (f"{family}_report.md", "report.html"):
        report_text = (reports / filename).read_text(encoding="utf-8")
        assert "Unverified projection of doubled profits" not in report_text
    assert all(record["disposition"] != "act" for record in delivered["recommendation_records"])
    outputs = {r["tool_name"]: r["output"] for r in raw["tool_results"] if r["status"] == "success"}
    answer: dict[str, Any] = outputs[tool]

    report_md = (reports / f"{family}_report.md").read_text(encoding="utf-8")

    if family == "survival":
        assert (answer["n_rows"], answer["n_events"], answer["survival_at_end"]) == (80, 40, 0.5)
    elif family == "basket":
        pair = next(r for r in answer["rules"] if {r["antecedent"], r["consequent"]} == {"tea", "milk"})
        assert (answer["n_orders"], pair["orders"], pair["lift"]) == (80, 20, 4.0)
    elif family == "network":
        assert answer["results"]["num_nodes"] == 8
        assert answer["results"]["top_hub"]["node"] == "hub"
    elif family == "geospatial":
        assert answer["n_points"] == 60
        assert abs(answer["centroid"]["lon"]) == pytest.approx(180.0)
    elif family == "survey":
        item = next(c for c in raw["data_profile"]["columns"] if c["name"] == "q1_service")
        assert item["stats"]["mean"] == 3.0
        assert answer["rows_after"] == 60
    elif family == "transactional":
        assert answer["transaction_count"] == 80
        assert answer["order_count"] == 40
        assert answer["total_revenue"] == 6600.0
        assert "transaction" in report_md.lower() or "order" in report_md.lower()
    elif family == "continuous_series":
        assert answer["periods_used"] == 60
        assert answer["trend_direction"] == "increasing"
        assert answer["sen_slope"] == 2.0
        assert "trend" in report_md.lower() or "increasing" in report_md.lower()
    elif family == "panel":
        assert answer["measure_column"] == "sales"
        assert answer["dimension_column"] == "store_id"
        assert "summary" in answer and len(answer["summary"]) > 0
    elif family == "cross_sectional":
        assert len(answer["comparisons"]) >= 1
        comp = answer["comparisons"][0]
        assert comp["n"] == 40 and comp["n_rest"] == 40
        assert "p_adjusted" in comp
    elif family == "hybrid_text":
        assert answer["non_null_count"] == 50
        assert answer["missing_pct"] == 0.0
        assert "and" not in answer["top_tokens"]
        assert "software" in answer["top_tokens"]


@pytest.mark.parametrize("family", ALL_ARCHETYPE_FAMILIES)
def test_archetype_positive_controls(family: str, tmp_path: Path) -> None:
    """Positive controls for all 10 archetype families with planted genuine signals.

    Directly asserts numerical evidence dictionary fields and verified prose.
    """
    if family == "transactional":
        # 1. Net revenue after returns, line items vs orders, customer activity leakage guard
        tx_df = pd.DataFrame({
            "order_id": [f"ORD_{i // 2:04d}" for i in range(100)],
            "customer_id": [f"CUST_{i % 10:02d}" for i in range(100)],
            "timestamp": [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(100)],
            "amount": [100.0 if i % 10 != 9 else -50.0 for i in range(100)],
            "quantity": [1] * 100,
        })
        tx_path = tmp_path / "tx_pos.csv"
        tx_df.to_csv(tx_path, index=False)
        tx_tool = CohortAnalysisTool()
        tx_out = tx_tool.execute(
            str(tx_path),
            date_column="timestamp",
            amount_column="amount",
            order_column="order_id",
            customer_column="customer_id",
        )
        assert tx_out["total_revenue"] == 8500.0
        assert tx_out["transaction_count"] == 100
        assert tx_out["order_count"] == 50
        assert tx_out["mean_lines_per_order"] == 2.0
        assert "8,500.00" in tx_out["summary"]

        reshaped_tx, notes = aggregate_point_in_time_event_features(tx_df, "customer_id", "timestamp")
        cust0 = reshaped_tx[reshaped_tx["customer_id"] == "CUST_00"].sort_values("timestamp")
        assert list(cust0["cust_prior_event_count"]) == list(range(10))
        assert (cust0["cust_lifetime_frequency"] == 10).all()
        assert any("cust_prior_event_count" in n for n in notes)

    elif family == "continuous_series":
        # 2. Demand change, forecast vs repeating last value baseline, temporal split
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        cs_df = pd.DataFrame({
            "timestamp": dates,
            "demand": [100.0 + 5.0 * i + (i % 7) for i in range(60)],
        })
        cs_path = tmp_path / "cs_pos.csv"
        cs_df.to_csv(cs_path, index=False)
        fc_tool = ForecastAnalysisTool()
        cs_out = fc_tool.execute(str(cs_path), date_column="timestamp", value_column="demand", horizon=7)
        assert cs_out["mase"] < 1.0
        assert cs_out["series"][0]["naive_no_better"] is False
        assert cs_out["series"][0]["backtest_periods"] > 0
        assert cs_out["series"][0]["forecast_end_value"] > cs_out["series"][0]["last_value"]
        assert "Best model:" in cs_out["summary"]

    elif family == "panel":
        # 3. Within-store trend vs store mix, lag leakage guard across entities
        rng = np.random.default_rng(42)
        p_dates = pd.date_range("2024-01-01", periods=12, freq="W")
        p_rows: list[dict[str, Any]] = []
        for i in range(6):
            store = f"Store_{i}"
            base = 2000.0 * (i + 1)
            for d in p_dates:
                p_rows.append({
                    "store_id": store,
                    "week_date": d,
                    "sales": float(base + 10.0 * np.sin(len(p_rows))),
                    "foot_traffic": float(100.0 + 5.0 * i + rng.normal(0, 1)),
                })
        p_df = pd.DataFrame(p_rows)
        p_path = tmp_path / "panel_pos.csv"
        p_df.to_csv(p_path, index=False)
        mm_tool = MixedModelAnalysisTool()
        p_out = mm_tool.execute(
            str(p_path),
            target_column="sales",
            group_column="store_id",
            feature_columns=["foot_traffic"],
        )
        assert p_out["icc"] > 0.80
        assert p_out["n_groups"] == 6
        assert "variation in sales is between" in p_out["summary"].lower()

        p_reshaped, p_notes = compute_panel_deltas_and_lags(p_df, entity_col="store_id", time_col="week_date")
        for store in [f"Store_{i}" for i in range(6)]:
            first_store_row = p_reshaped[p_reshaped["store_id"] == store].iloc[0]
            assert pd.isna(first_store_row["sales_lag1"])
        assert any("sales_lag1" in n for n in p_notes)

    elif family == "cross_sectional":
        # 4. Segment comparison, group contrast denominator, multiple testing correction
        rng = np.random.default_rng(42)
        cs_rows: list[dict[str, Any]] = []
        for i in range(100):
            dept = "Engineering" if i < 25 else "Sales" if i < 50 else "Marketing" if i < 75 else "Support"
            base = 150000.0 if dept == "Engineering" else 80000.0
            salary = float(base + rng.normal(0, 5000))
            cs_rows.append({"emp_id": f"E{i}", "department": dept, "salary": salary})
        cs_cross_df = pd.DataFrame(cs_rows)
        cs_cross_path = tmp_path / "cross_pos.csv"
        cs_cross_df.to_csv(cs_cross_path, index=False)
        seg_tool = SegmentComparisonTool()
        cross_out = seg_tool.execute(str(cs_cross_path), measure_column="salary", dimension_column="department")
        eng = next(c for c in cross_out["comparisons"] if c["level"] == "Engineering")
        assert eng["lift"] > 0.40
        assert eng["n"] == 25 and eng["n_rest"] == 75
        assert eng["p_adjusted"] is not None and eng["p_adjusted"] < 0.001
        assert "Engineering" in cross_out["summary"]

    elif family == "survival":
        # 5. Event-free share at specified horizon, time-zero risk set
        surv_df = pd.DataFrame({
            "account_id": [f"A{i}" for i in range(100)],
            "tenure": [0.0] * 5 + [5.0] * 20 + [10.0] * 25 + [20.0] * 50,
            "churn": [1] * 5 + [1] * 20 + [1] * 25 + [0] * 50,
        })
        surv_path = tmp_path / "surv_pos.csv"
        surv_df.to_csv(surv_path, index=False)
        surv_tool = SurvivalAnalysisTool()
        surv_out = surv_tool.execute(str(surv_path), duration_column="tenure", event_column="churn")
        assert surv_out["n_rows"] == 100
        assert surv_out["n_events"] == 50
        assert surv_out["n_censored"] == 50
        assert surv_out["survival_at_end"] == 0.50
        assert len(surv_out["milestones"]) > 0
        assert "median survival" in surv_out["summary"].lower() or "50%" in surv_out["summary"]

    elif family == "survey":
        # 6. Items driving score, reverse-scoring, declared endpoints, Cronbach's alpha
        survey_df = pd.DataFrame({
            "resp_id": [f"R{i}" for i in range(60)],
            "q1_helpful": np.tile([3, 4, 5, 4, 5], 12),
            "q2_clear": np.tile([3, 4, 5, 4, 5], 12),
            "q3_satisfied": np.tile([3, 4, 5, 4, 5], 12),
            "q4_difficult": np.tile([3, 2, 1, 2, 1], 12),
        })
        survey_reshaped, surv_notes = assess_survey_reliability_and_polarity(survey_df, declared_endpoints=(1.0, 5.0))
        assert "q4_difficult_rev" in survey_reshaped.columns
        assert (survey_reshaped["q4_difficult_rev"] + survey_df["q4_difficult"] == 6.0).all()
        assert any("Cronbach's alpha" in n for n in surv_notes)
        assert any("complete responses" in n for n in surv_notes)

    elif family == "basket":
        # 7. Co-occurrence vs base rates, order support denominator, lift phrasing
        b_rows: list[tuple[str, str]] = []
        for i in range(80):
            b_id = f"B{i}"
            if i < 20:
                b_rows.extend([(b_id, "coffee"), (b_id, "sugar")])
            else:
                b_rows.extend([(b_id, "bread"), (b_id, "butter")])
        b_df = pd.DataFrame(b_rows, columns=["basket_id", "item_name"])
        b_path = tmp_path / "basket_pos.csv"
        b_df.to_csv(b_path, index=False)
        bsk_tool = BasketAnalysisTool()
        b_out = bsk_tool.execute(str(b_path), order_column="basket_id", item_column="item_name")
        assert b_out["n_orders"] == 80
        b_rule = next(r for r in b_out["rules"] if r["antecedent"] == "coffee" and r["consequent"] == "sugar")
        assert b_rule["orders"] == 20
        assert b_rule["lift"] == 4.0
        b_finds = bsk_tool.findings(b_out, None, None)
        assert len(b_finds) > 0
        assert "times its share across all orders" in b_finds[0].headline

    elif family == "network":
        # 8. Reach and influence, degree vs PageRank separation, sink mass conservation
        net_rows: list[dict[str, Any]] = []
        for i in range(10):
            net_rows.append({"source": f"D{i}", "target": "hub", "weight": 1.0})
        net_rows.append({"source": "whale_src", "target": "whale", "weight": 1000.0})
        net_rows.append({"source": "hub", "target": "sink_auth", "weight": 10.0})
        net_rows.append({"source": "whale", "target": "sink_auth", "weight": 1000.0})

        net_df = pd.DataFrame(net_rows)
        net_path = tmp_path / "net_pos.csv"
        net_df.to_csv(net_path, index=False)
        net_tool = GraphAnalysisTool()
        net_out = net_tool.execute(str(net_path), source_column="source", target_column="target", weight_column="weight")
        assert net_out["results"]["top_hub"]["node"] == "hub"
        assert net_out["results"]["top_strength"]["node"] == "whale"
        assert net_out["results"]["top_pagerank"]["node"] == "sink_auth"
        assert net_out["results"]["top_hub"]["node"] != net_out["results"]["top_strength"]["node"]
        assert net_out["results"]["top_hub"]["node"] != net_out["results"]["top_pagerank"]["node"]
        assert net_out["results"]["pagerank_mass_sum"] == pytest.approx(1.0, abs=1e-5)
        assert "sink_auth" in net_out["summary"] or "hub" in net_out["summary"]

    elif family == "geospatial":
        # 9. Demand concentration vs population, date-line spherical distance, invalid coordinates
        geo_df = pd.DataFrame({
            "ride_id": [f"R{i}" for i in range(65)],
            "lat": [*np.tile([10.0, 11.0], 30).tolist(), 120.0, 150.0, 95.0, 100.0, 110.0],
            "lon": [*np.tile([-179.5, 179.5], 30).tolist(), 200.0, -250.0, 300.0, -190.0, 185.0],
        })
        geo_path = tmp_path / "geo_pos.csv"
        geo_df.to_csv(geo_path, index=False)
        geo_tool = GeospatialAnalysisTool()
        geo_out = geo_tool.execute(str(geo_path), lat_column="lat", lon_column="lon")
        assert geo_out["n_excluded"] == 5
        assert abs(geo_out["centroid"]["lon"]) == pytest.approx(180.0, abs=2.0)
        assert geo_out["n_points"] == 60
        assert len(geo_out["densest_cells"]) > 0

    elif family == "hybrid_text":
        # 10. Themes vs stopwords, coverage disclosure, non-causal attribution
        text_comments = [
            "The software architecture is well thought out, fast and highly extensible for data science.",
            "Exceptional software performance, smooth animations, and clean type-safe python implementation.",
            "Robust architecture with comprehensive test suites and clear design patterns.",
        ]
        txt_df = pd.DataFrame({
            "id": [f"ID_{i}" for i in range(60)],
            "review_text": [text_comments[i % len(text_comments)] for i in range(60)],
        })
        txt_path = tmp_path / "text_pos.csv"
        txt_df.to_csv(txt_path, index=False)
        txt_tool = TextAnalysisTool()
        txt_out = txt_tool.execute(str(txt_path), text_column="review_text")
        assert txt_out["non_null_count"] == 60
        assert txt_out["missing_pct"] == 0.0
        assert "software" in txt_out["top_tokens"]
        assert "architecture" in txt_out["top_tokens"]
        assert "and" not in txt_out["top_tokens"]
        assert "the" not in txt_out["top_tokens"]
        assert "text column 'review_text'" in txt_out["summary"].lower()


@pytest.mark.parametrize("family", ALL_ARCHETYPE_FAMILIES)
def test_archetype_negative_unanswerable_controls(family: str, tmp_path: Path) -> None:
    """Negative and unanswerable controls for all 10 archetype families.

    Verifies inconclusive/unperformed checks, boundary rejection, zero fabricated metrics,
    and appropriate caveats.
    """
    rng = np.random.default_rng(0)

    if family == "transactional":
        # Unparseable transaction dates must raise ToolExecutionError, zero fabricated revenue
        bad_tx_df = pd.DataFrame({
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
            "timestamp": ["invalid_date", "not_a_date"],
            "amount": [10.0, 20.0],
        })
        bad_tx_path = tmp_path / "bad_tx.csv"
        bad_tx_df.to_csv(bad_tx_path, index=False)
        with pytest.raises(ToolExecutionError, match="No rows have both a parseable date and a numeric amount"):
            CohortAnalysisTool().execute(str(bad_tx_path), date_column="timestamp", amount_column="amount")

    elif family == "continuous_series":
        # Unpredictable random walk yields naive_no_better is True, MASE >= 1.0
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        rw = np.cumsum(rng.normal(0, 10, size=40))
        noise_df = pd.DataFrame({"timestamp": dates, "demand": rw})
        noise_path = tmp_path / "noise_series.csv"
        noise_df.to_csv(noise_path, index=False)
        noise_out = ForecastAnalysisTool().execute(str(noise_path), date_column="timestamp", value_column="demand", horizon=7)
        assert noise_out["series"][0]["naive_no_better"] is True
        assert noise_out["mase"] >= 1.0

    elif family == "panel":
        # Identical store sales with zero between-store variance (ICC <= 0.05)
        p_dates = pd.date_range("2024-01-01", periods=12, freq="W")
        p_rows: list[dict[str, Any]] = []
        for i in range(6):
            store = f"Store_{i}"
            for d in p_dates:
                p_rows.append({
                    "store_id": store,
                    "week_date": d,
                    "sales": float(500.0 + rng.normal(0, 10)),
                    "foot_traffic": float(100.0 + rng.normal(0, 5)),
                })
        p_df = pd.DataFrame(p_rows)
        p_path = tmp_path / "noise_panel.csv"
        p_df.to_csv(p_path, index=False)
        p_out = MixedModelAnalysisTool().execute(
            str(p_path),
            target_column="sales",
            group_column="store_id",
            feature_columns=["foot_traffic"],
        )
        assert p_out["icc"] <= 0.05

    elif family == "cross_sectional":
        # Completely identical salary distributions across departments -> zero significant p_adjusted
        cs_rows: list[dict[str, Any]] = []
        for i in range(100):
            dept = "Engineering" if i < 25 else "Sales" if i < 50 else "Marketing" if i < 75 else "Support"
            salary = float(100000.0 + rng.normal(0, 5000))
            cs_rows.append({"emp_id": f"E{i}", "department": dept, "salary": salary})
        cs_df = pd.DataFrame(cs_rows)
        cs_path = tmp_path / "noise_cross.csv"
        cs_df.to_csv(cs_path, index=False)
        cross_out = SegmentComparisonTool().execute(str(cs_path), measure_column="salary", dimension_column="department")
        p_adjs = [c["p_adjusted"] for c in cross_out["comparisons"] if c["p_adjusted"] is not None]
        assert all(p > 0.05 for p in p_adjs)

    elif family == "survival":
        # 100% censoring raises ToolExecutionError; negative durations excluded in cleaning
        surv_df = pd.DataFrame({
            "account_id": [f"A{i}" for i in range(50)],
            "tenure": [10.0] * 50,
            "churn": [0] * 50,
        })
        surv_path = tmp_path / "censored_surv.csv"
        surv_df.to_csv(surv_path, index=False)
        with pytest.raises(ToolExecutionError, match="has no events after coding"):
            SurvivalAnalysisTool().execute(str(surv_path), duration_column="tenure", event_column="churn")

        corrupt_surv = pd.DataFrame({"tenure": [-5.0, 10.0], "churn": [1, 0]})
        clean_surv, notes = verify_and_prepare_survival_data(corrupt_surv, duration_col="tenure", event_col="churn")
        assert len(clean_surv) == 1
        assert any("negative duration" in n.lower() or "excluded" in n.lower() for n in notes)

    elif family == "survey":
        # Single column scale cannot compute Cronbach's alpha
        single_survey = pd.DataFrame({"q1": [1, 2, 3, 4, 5]})
        _, notes = assess_survey_reliability_and_polarity(single_survey)
        assert len(notes) == 0

    elif family == "basket":
        # Independent random item assignment produces zero high-lift association rules
        b_rows: list[tuple[str, str]] = []
        for i in range(80):
            b_id = f"B{i}"
            items = [itm for itm in ["coffee", "tea", "milk", "sugar", "bread"] if rng.uniform() > 0.6]
            for itm in items:
                b_rows.append((b_id, itm))
        b_df = pd.DataFrame(b_rows, columns=["basket_id", "item_name"])
        b_path = tmp_path / "noise_basket.csv"
        b_df.to_csv(b_path, index=False)
        b_out = BasketAnalysisTool().execute(str(b_path), order_column="basket_id", item_column="item_name")
        high_lift_rules = [r for r in b_out["rules"] if r["lift"] > 2.0]
        assert len(high_lift_rules) == 0

    elif family == "network":
        # Empty graph raises ToolExecutionError
        empty_df = pd.DataFrame({"source": [], "target": []})
        empty_path = tmp_path / "empty_net.csv"
        empty_df.to_csv(empty_path, index=False)
        with pytest.raises(ToolExecutionError):
            GraphAnalysisTool().execute(str(empty_path), source_column="source", target_column="target")

    elif family == "geospatial":
        # Antipodal points produce undefined spherical center (None) without crashing
        antipodal_lat = np.array([45.0, -45.0])
        antipodal_lon = np.array([0.0, 180.0])
        assert spherical_center(antipodal_lat, antipodal_lon) is None

    elif family == "hybrid_text":
        # Pure stopword reviews yield vocab_size 0 and empty top_tokens
        stopword_df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "review_text": ["the and of in", "for with at by", "to from on as"],
        })
        stop_path = tmp_path / "stopwords.csv"
        stopword_df.to_csv(stop_path, index=False)
        stop_out = TextAnalysisTool().execute(str(stop_path), text_column="review_text")
        assert stop_out["vocab_size"] == 0
        assert len(stop_out["top_tokens"]) == 0
