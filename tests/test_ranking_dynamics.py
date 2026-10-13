from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
        entity_a_col="home_team",
        entity_b_col="away_team",
        score_a_col="home_goals",
        score_b_col="away_goals",
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
    assert f.chart_hint == "bar"
    assert "arsenal" in f.headline.lower()
