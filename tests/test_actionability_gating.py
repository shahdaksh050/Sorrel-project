from __future__ import annotations

from src.core.findings import Finding, _is_actionable_feature, classify_action_matrix


def test_is_actionable_feature() -> None:
    # Controllable business levers: YES
    assert _is_actionable_feature("price") is True
    assert _is_actionable_feature("discount_rate") is True
    assert _is_actionable_feature("marketing_spend") is True
    assert _is_actionable_feature("staffing_level") is True
    assert _is_actionable_feature("contract_type") is True

    # External market odds & betting quotes: NO
    assert _is_actionable_feature("B365H") is False
    assert _is_actionable_feature("BWH") is False
    assert _is_actionable_feature("IWH") is False
    assert _is_actionable_feature("odds_home") is False
    assert _is_actionable_feature("bet_probability") is False
    assert _is_actionable_feature("ticker_close") is False

    # Surrogate IDs and keys: NO
    assert _is_actionable_feature("leagueID") is False
    assert _is_actionable_feature("customer_id") is False
    assert _is_actionable_feature("order_key") is False
    assert _is_actionable_feature("session_guid") is False
    assert _is_actionable_feature("index") is False

    # System timestamps: NO
    assert _is_actionable_feature("created_at") is False
    assert _is_actionable_feature("match_date") is False
    assert _is_actionable_feature("timestamp") is False

    # Weather: NO
    assert _is_actionable_feature("temperature") is False
    assert _is_actionable_feature("humidity") is False


def test_action_matrix_excludes_non_controllables() -> None:
    # Finding on external market odds
    odds_finding = Finding(
        finding_id="corr_b365h_bwh",
        kind="correlation",
        headline="B365H and BWH move together (r=0.99)",
        measure="B365H",
        dimension="BWH",
        effect=0.99,
        effect_kind="r",
        confidence=0.95,
        importance=0.8,
    )

    # Finding on controllable lever
    pricing_finding = Finding(
        finding_id="segment_tier_spend",
        kind="segment_lift",
        headline="Enterprise customers spend 2.4x more (lift=+140%)",
        measure="spend",
        dimension="subscription_tier",
        level="Enterprise",
        effect=1.4,
        effect_kind="lift",
        confidence=0.9,
        importance=0.9,
    )

    matrix = classify_action_matrix([odds_finding, pricing_finding])

    # Odds finding MUST NOT be in quick_wins or strategic_bets
    quick_win_ids = [item["finding_id"] for item in matrix["quick_wins"]]
    assert "corr_b365h_bwh" not in quick_win_ids
    assert "segment_tier_spend" in quick_win_ids

    # Odds finding routes to watch_caveats as external risk / covariate
    caveat_ids = [item["finding_id"] for item in matrix["watch_caveats"]]
    assert "corr_b365h_bwh" in caveat_ids
