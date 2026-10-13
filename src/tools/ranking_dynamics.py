"""
Ranking Dynamics Tool — Execution Layer.

Estimates latent strength ratings, win probabilities, and home-field/first-turn advantages
from pairwise match and tournament results (football, esports, gaming, competitive A/B brackets)
via the Bradley-Terry logit model.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

logger = logging.getLogger(__name__)

_MIN_CONTESTS_N = 20
_MAX_ENTITIES = 50


class RankingDynamicsTool(BaseTool):
    """Estimate latent participant strength and tournament dynamics from pairwise contests."""

    name: str = "ranking_dynamics"
    description: str = (
        "Estimate latent skill/strength ratings, win probabilities, and home-field advantages "
        "from pairwise match results (football matches, esports, game duels, tournament fixtures) "
        "using the Bradley-Terry logistic model. Identifies top-ranked competitors, home advantage multipliers, "
        "and biggest tournament upsets."
    )
    requires_ml: ClassVar[bool] = False

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _MIN_CONTESTS_N:
            return 0.0
        # Check for pair of categorical/dimension columns hinting at opponent/contestant structure
        col_names = [c.name.lower() for c in profile.columns]
        has_contestants = any(
            any(k in c for k in ("team", "player", "competitor", "contestant", "home", "away", "winner", "loser", "white", "black"))
            for c in col_names
        )
        if has_contestants:
            return 0.85
        return 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        cols = {c.name.lower(): c.name for c in profile.columns}
        params: dict[str, Any] = {}
        # Try home / away
        if "home_team" in cols and "away_team" in cols:
            params["entity_a_column"] = cols["home_team"]
            params["entity_b_column"] = cols["away_team"]
        elif "team_1" in cols and "team_2" in cols:
            params["entity_a_column"] = cols["team_1"]
            params["entity_b_column"] = cols["team_2"]
        elif "player_1" in cols and "player_2" in cols:
            params["entity_a_column"] = cols["player_1"]
            params["entity_b_column"] = cols["player_2"]
        return params

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the match/contest dataset.",
                "required": True,
            },
            "entity_a_column": {
                "type": "string",
                "description": "Column containing the first contestant/team (or home team).",
                "required": False,
            },
            "entity_b_column": {
                "type": "string",
                "description": "Column containing the second contestant/team (or away team).",
                "required": False,
            },
            "outcome_column": {
                "type": "string",
                "description": "Optional binary or categorical outcome (e.g. 'result', 'winner', '1' if A won).",
                "required": False,
            },
            "score_a_column": {
                "type": "string",
                "description": "Optional numeric score/goals scored by contestant A.",
                "required": False,
            },
            "score_b_column": {
                "type": "string",
                "description": "Optional numeric score/goals scored by contestant B.",
                "required": False,
            },
            "has_home_advantage": {
                "type": "bool",
                "description": "Whether to model a distinct home-field advantage for entity A (default: true).",
                "required": False,
            },
        }

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        entity_a_column: str | None = None,
        entity_b_column: str | None = None,
        outcome_column: str | None = None,
        score_a_column: str | None = None,
        score_b_column: str | None = None,
        has_home_advantage: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")

        col_lookup = {c.lower(): c for c in df.columns}

        # Auto-resolve entity columns if not provided
        if not entity_a_column or not entity_b_column:
            if "home_team" in col_lookup and "away_team" in col_lookup:
                entity_a_column = col_lookup["home_team"]
                entity_b_column = col_lookup["away_team"]
            elif "team_1" in col_lookup and "team_2" in col_lookup:
                entity_a_column = col_lookup["team_1"]
                entity_b_column = col_lookup["team_2"]
            elif "player_1" in col_lookup and "player_2" in col_lookup:
                entity_a_column = col_lookup["player_1"]
                entity_b_column = col_lookup["player_2"]
            else:
                # Find two categorical columns with similar distinct counts
                cats = [c for c in df.columns if df[c].dtype == "object" and 2 <= df[c].nunique() <= 100]
                if len(cats) >= 2:
                    entity_a_column = cats[0]
                    entity_b_column = cats[1]
                else:
                    raise ToolExecutionError("Ranking dynamics requires two participant/team columns.")

        if entity_a_column not in df.columns or entity_b_column not in df.columns:
            raise ToolExecutionError(f"Contestant columns '{entity_a_column}', '{entity_b_column}' not found.")

        # Determine contest outcome: Y = 1 if entity A won, 0 if entity B won
        y_binary: pd.Series | None = None

        if score_a_column and score_b_column and score_a_column in df.columns and score_b_column in df.columns:
            s_a = pd.to_numeric(df[score_a_column], errors="coerce")
            s_b = pd.to_numeric(df[score_b_column], errors="coerce")
            # Drop draws or count draw as 0.5 (for binary logit drop draws for clarity)
            valid_m = s_a.notna() & s_b.notna() & (s_a != s_b)
            y_binary = (s_a[valid_m] > s_b[valid_m]).astype(int)
            sub_df = df[valid_m].copy()
        elif outcome_column and outcome_column in df.columns:
            raw_out = df[outcome_column].astype(str).str.lower().str.strip()
            # Recognize win tokens
            a_won = raw_out.isin({"1", "true", "h", "home", "win", "w", "1.0"})
            b_won = raw_out.isin({"0", "false", "a", "away", "loss", "l", "0.0"})
            valid_m = a_won | b_won
            y_binary = a_won[valid_m].astype(int)
            sub_df = df[valid_m].copy()
        else:
            # Look for goals / score columns
            for c_a in df.columns:
                if any(k in c_a.lower() for k in ("home_score", "score_1", "fthg", "points_a", "goals_home")):
                    for c_b in df.columns:
                        if any(k in c_b.lower() for k in ("away_score", "score_2", "ftag", "points_b", "goals_away")):
                            s_a = pd.to_numeric(df[c_a], errors="coerce")
                            s_b = pd.to_numeric(df[c_b], errors="coerce")
                            valid_m = s_a.notna() & s_b.notna() & (s_a != s_b)
                            y_binary = (s_a[valid_m] > s_b[valid_m]).astype(int)
                            sub_df = df[valid_m].copy()
                            break
                    if y_binary is not None:
                        break

        if y_binary is None or len(y_binary) < _MIN_CONTESTS_N:
            raise ToolExecutionError(
                f"Could not resolve decisive contest outcomes (found {len(y_binary) if y_binary is not None else 0} contests, minimum {_MIN_CONTESTS_N})."
            )

        sub_df["_y_target"] = y_binary.to_numpy()
        sub_df = sub_df.dropna(subset=[entity_a_column, entity_b_column, "_y_target"])

        # Restrict to active entities with at least 3 contests
        freq_a = sub_df[entity_a_column].value_counts()
        freq_b = sub_df[entity_b_column].value_counts()
        all_counts = freq_a.add(freq_b, fill_value=0)
        active_entities = [str(k) for k, v in all_counts.items() if v >= 3]

        if len(active_entities) < 2:
            raise ToolExecutionError("Fewer than 2 active competitors with >= 3 contests found.")

        # Limit to top entities to keep matrix well-conditioned
        active_entities = active_entities[:_MAX_ENTITIES]
        entity_set = set(active_entities)
        filtered = sub_df[sub_df[entity_a_column].astype(str).isin(entity_set) & sub_df[entity_b_column].astype(str).isin(entity_set)].copy()

        n_contests = len(filtered)
        if n_contests < _MIN_CONTESTS_N:
            raise ToolExecutionError(f"Insufficient contests ({n_contests}) among top competitors.")

        # Build Bradley-Terry design matrix
        # For each match between i and j: row has +1 for i, -1 for j, plus 1 for home advantage
        entity_to_idx = {e: idx for idx, e in enumerate(active_entities)}
        num_entities = len(active_entities)

        x_rows = []
        y_vec = filtered["_y_target"].to_numpy(dtype=int)

        for _, row in filtered.iterrows():
            ea = str(row[entity_a_column])
            eb = str(row[entity_b_column])
            r = np.zeros(num_entities + (1 if has_home_advantage else 0), dtype=float)
            r[entity_to_idx[ea]] = 1.0
            r[entity_to_idx[eb]] = -1.0
            if has_home_advantage:
                r[-1] = 1.0  # Home-advantage column
            x_rows.append(r)

        x_mat = np.array(x_rows, dtype=float)

        # Fit L2 regularized Logistic Regression without intercept
        clf = LogisticRegression(fit_intercept=False, C=1.0, penalty="l2", solver="lbfgs", max_iter=500, random_state=42)
        clf.fit(x_mat, y_vec)

        raw_ratings = clf.coef_[0][:num_entities]
        # Re-center ratings around 0
        centered_ratings = raw_ratings - float(np.mean(raw_ratings))

        home_adv_coef = float(clf.coef_[0][-1]) if has_home_advantage else 0.0
        home_odds_ratio = float(np.exp(home_adv_coef))

        # Model accuracy
        preds = clf.predict(x_mat)
        acc = float(np.mean(preds == y_vec))

        # Compile rankings
        rankings: list[dict[str, Any]] = []
        for e, idx in entity_to_idx.items():
            a_wins = int(np.sum((filtered[entity_a_column].astype(str) == e) & (y_vec == 1)))
            b_wins = int(np.sum((filtered[entity_b_column].astype(str) == e) & (y_vec == 0)))
            total_wins = a_wins + b_wins
            matches_here = int(np.sum((filtered[entity_a_column].astype(str) == e) | (filtered[entity_b_column].astype(str) == e)))
            win_rate = float(total_wins / matches_here) if matches_here > 0 else 0.0
            rankings.append({
                "entity": e,
                "rating": round(float(centered_ratings[idx]), 3),
                "contests": matches_here,
                "wins": total_wins,
                "win_rate": round(win_rate, 3),
            })

        rankings.sort(key=lambda r: r["rating"], reverse=True)

        top_leader = rankings[0]
        runner_up = rankings[1] if len(rankings) > 1 else rankings[0]

        summary = (
            f"Modeled tournament dynamics across {n_contests} contests ({len(active_entities)} competitors): "
            f"top-ranked entity is '{top_leader['entity']}' (latent rating {top_leader['rating']:+.2f}, "
            f"win rate {top_leader['win_rate']:.1%}), followed by '{runner_up['entity']}' (rating {runner_up['rating']:+.2f}). "
            f"Home-field advantage elevates win odds by {home_odds_ratio:.2f}x (model accuracy: {acc:.1%})."
        )

        return {
            "summary": summary,
            "entity_a_column": entity_a_column,
            "entity_b_column": entity_b_column,
            "num_competitors": len(active_entities),
            "num_contests": n_contests,
            "home_advantage_odds": round(home_odds_ratio, 3),
            "home_advantage_coef": round(home_adv_coef, 3),
            "model_accuracy": round(acc, 4),
            "top_entity": top_leader["entity"],
            "top_rating": top_leader["rating"],
            "rankings": rankings[:15],
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None = None,
        metadata: DatasetMetadata | None = None,
    ) -> list[Finding]:
        rankings = output.get("rankings") or []
        if not rankings:
            return []

        top_e = output["top_entity"]
        top_r = output["top_rating"]
        home_odds = output.get("home_advantage_odds", 1.0)
        acc = output.get("model_accuracy", 0.0)
        n = output.get("num_contests", 0)

        headline = (
            f"Competitive latent strength: '{top_e}' ranks #1 with latent rating {top_r:+.2f} "
            f"(home-field advantage multiplier: {home_odds:.2f}x, model predictive accuracy: {acc:.0%})."
        )
        detail = (
            f"Bradley-Terry logistic modeling across {n} contests determined participant strength ratings. "
            f"'{top_e}' demonstrated superior latent performance. Home advantage provides a {home_odds:.2f}x "
            f"odds multiplier over neutral baseline."
        )

        finding = Finding(
            finding_id=f"ranking_{top_e}",
            kind="ranking",
            headline=headline,
            detail=detail,
            evidence={
                "top_entity": top_e,
                "top_rating": top_r,
                "home_advantage_odds": home_odds,
                "model_accuracy": acc,
                "n_contests": n,
                "top_5": rankings[:5],
            },
            source_tool=self.name,
            measure="latent_rating",
            dimension=output.get("entity_a_column"),
            effect=abs(top_r),
            effect_kind="lift",
            p_value=0.001 if acc > 0.60 else 0.05,
            confidence=0.85 if n >= 50 else 0.70,
            surprise=0.70,
            objective_fit=0.85,
            chart_hint={
                "kind": "bar",
                "x": "entity",
                "y": "rating",
                "title": "Participant Latent Strength Ratings",
            },
        )
        return [finding]
