from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.tools.ranking_dynamics import RankingDynamicsTool


def test_ranking_dynamics_bradley_terry_ratings(tmp_path: Path) -> None:
    # 4 teams: Arsenal (strongest), Chelsea (strong), Everton (average), Norwich (weak)
    # Generate pairwise round-robin matches
    teams = ["Arsenal", "Chelsea", "Everton", "Norwich"]
    true_skills = {"Arsenal": 2.0, "Chelsea": 1.0, "Everton": 0.0, "Norwich": -1.5}
    home_adv = 0.4

    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for _ in range(30):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                # Logit probability: P(H wins) = logit_inv(skill_H - skill_A + home_adv)
                logit_p = true_skills[h] - true_skills[a] + home_adv
                prob_h = 1.0 / (1.0 + np.exp(-logit_p))
                h_goals = rng.poisson(max(0.5, 1.5 * prob_h * 2))
                a_goals = rng.poisson(max(0.5, 1.5 * (1 - prob_h) * 2))
                rows.append({"home_team": h, "away_team": a, "home_goals": h_goals, "away_goals": a_goals})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "league_matches.csv"
    df.to_csv(csv_path, index=False)

    tool = RankingDynamicsTool()
    res = tool.run(
        file_path=str(csv_path),
        entity_a_column="home_team",
        entity_b_column="away_team",
        score_a_column="home_goals",
        score_b_column="away_goals",
    )
    assert res.status == "success"
    out: dict[str, Any] = res.output

    assert "ratings" in out
    ratings = out["ratings"]
    assert len(ratings) == 4
    # Top rated team should be Arsenal
    assert ratings[0]["entity"] == "Arsenal"
    # Lowest rated team should be Norwich
    assert ratings[-1]["entity"] == "Norwich"
    assert out["home_advantage_odds_ratio"] >= 1.0

    findings = tool.findings(out)
    assert len(findings) >= 1
    f = findings[0]
    assert f.kind == "ranking"
    assert f.chart_hint["kind"] == "bar"
    assert "arsenal" in f.headline.lower()


def test_ranking_dynamics_accepts_explicit_deprecated_column_aliases(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for _ in range(9):
        rows.extend([
            {"a": "A", "b": "B", "a_score": 2, "b_score": 0},
            {"a": "A", "b": "B", "a_score": 1, "b_score": 0},
            {"a": "A", "b": "B", "a_score": 0, "b_score": 1},
        ])
    csv_path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = RankingDynamicsTool().run(
        file_path=str(csv_path),
        entity_a_col="a",
        entity_b_col="b",
        score_a_col="a_score",
        score_b_col="b_score",
    )

    assert result.status == "success"
    assert result.output["ratings"][0]["entity"] == "A"


def test_ranking_dynamics_rejects_unknown_aliases() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        RankingDynamicsTool().execute(file_path="not-read.csv", entity_left_col="team_a")
