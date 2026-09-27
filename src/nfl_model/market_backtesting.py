"""Chronological discrete-grid pricing using previously validated OOT projections."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd

from nfl_model.data_quality import require_columns, require_unique_keys
from nfl_model.discrete_scores import DiscreteGameModel, DiscreteScoreDistribution, ScoreGrid
from nfl_model.distribution_backtesting import MODEL_NAMES
from nfl_model.probability_evaluation import summarize_probabilities
from nfl_model.settlement import SettlementRules, grade_market, no_vig_probability, overtime_format, price_comparison, validate_scores


def market_specs(row) -> list[dict]:
    """Translate nflverse's home-favored spread convention exactly once."""
    specs = []
    for side, other in (("home", "away"), ("away", "home")):
        specs.append({"market": "moneyline", "side": side, "line": None,
                      "odds": getattr(row, f"market_{side}_moneyline"), "opposite_odds": getattr(row, f"market_{other}_moneyline"), "primary_side": side == "home"})
        if pd.notna(row.market_spread_line):
            specs.append({"market": "spread", "side": side, "line": -float(row.market_spread_line),
                          "odds": getattr(row, f"market_{side}_spread_odds"), "opposite_odds": getattr(row, f"market_{other}_spread_odds"), "primary_side": side == "home"})
    if pd.notna(row.market_total_line):
        for side, other in (("over", "under"), ("under", "over")):
            specs.append({"market": "total", "side": side, "line": float(row.market_total_line),
                          "odds": getattr(row, f"market_{side}_odds"), "opposite_odds": getattr(row, f"market_{other}_odds"), "primary_side": side == "over"})
    return specs


def price_game(row, grid: ScoreGrid, *, model: str, phase: str, rules: SettlementRules) -> list[dict]:
    output = []
    format_info = overtime_format(row.season, row.game_type)
    for spec in market_specs(row):
        args = {key: spec[key] for key in ("market", "side", "line")}
        probabilities = grid.market_probabilities(**args, rules=rules)
        decision = probabilities["p_win"] + probabilities["p_loss"]
        comparison = {}
        if pd.notna(spec["odds"]):
            exhaustive_pair = spec["market"] != "moneyline" or rules.moneyline_tie == "push"
            opposite = spec["opposite_odds"] if exhaustive_pair and pd.notna(spec["opposite_odds"]) else None
            comparison = price_comparison(**probabilities, odds=spec["odds"], opposite_odds=opposite)
        output.append({"game_id": row.game_id, "season": row.season, "gameday": row.gameday,
                       "home_team": row.home_team, "away_team": row.away_team, "game_type": row.game_type,
                       "model": model, "phase": phase, **spec, **probabilities,
                       "p_conditional": probabilities["p_win"] / decision if decision else np.nan,
                       "outcome": int(grade_market(row.home_score, row.away_score, **args, rules=rules)),
                       "overtime_regime": format_info["regime"], "moneyline_tie_rule": rules.moneyline_tie,
                       "includes_overtime": True, **comparison})
    return output


@dataclass
class MarketProbabilityRun:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    calibration_bins: pd.DataFrame
    fit_log: pd.DataFrame
    models: dict[str, DiscreteGameModel]
    metadata: dict


def run_market_backtest(
    games: pd.DataFrame, residuals: pd.DataFrame, *, mean_models: dict | None = None,
    first_evaluation_season: int = 2023, audit_season: int = 2025,
    rules: SettlementRules = SettlementRules(), progress: Callable[[str], None] | None = None,
) -> MarketProbabilityRun:
    rules.validate()
    if first_evaluation_season >= audit_season:
        raise ValueError("Need development seasons preceding the retrospective audit")
    require_unique_keys(games, ["game_id"], "game dataset")
    require_unique_keys(residuals, ["game_id", "model"], "out-of-time predictions")
    require_columns(games, ["season", "gameday", "game_type", "home_score", "away_score"], "game dataset")
    require_columns(residuals, ["season", "gameday", "mean_fit_end", "pred_home_score", "pred_away_score", "residual_home_score", "residual_away_score"], "out-of-time predictions")
    data, errors = games.copy(), residuals.copy()
    data["gameday"] = pd.to_datetime(data.gameday)
    errors["gameday"] = pd.to_datetime(errors.gameday)
    errors["mean_fit_end"] = pd.to_datetime(errors.mean_fit_end)
    if data[["gameday", "season"]].isna().any().any():
        raise ValueError("Game dates and seasons must not be missing")
    validate_scores(data.home_score, data.away_score)
    for row in data.itertuples(index=False):
        if not overtime_format(row.season, row.game_type)["final_tie_allowed"] and row.home_score == row.away_score:
            raise ValueError("A postseason final cannot be tied")
    if errors[["gameday", "mean_fit_end"]].isna().any().any() or not errors.mean_fit_end.lt(errors.gameday).all():
        raise ValueError("Mean projections must be strictly out of time")
    identity = errors.merge(data[["game_id", "gameday", "season"]], on="game_id", how="left", suffixes=("", "_history"), validate="many_to_one")
    if not identity.gameday.eq(identity.gameday_history).all() or not identity.season.eq(identity.season_history).all():
        raise ValueError("Cached prediction dates/seasons disagree with the game dataset")
    predictions, fit_logs, final_models = [], [], {}
    for season in range(first_evaluation_season, audit_season + 1):
        history = data.loc[data.season.lt(season)].sort_values(["gameday", "game_id"])
        forecast = data.loc[data.season.eq(season)].sort_values(["gameday", "game_id"])
        if history.empty or forecast.empty or not history.gameday.max() < forecast.gameday.min():
            raise ValueError(f"Missing or overlapping chronological data for {season}")
        phase = "retrospective_audit" if season == audit_season else "development"
        season_rows = []
        for name in MODEL_NAMES:
            calibration = errors.loc[errors.model.eq(name) & errors.season.lt(season)]
            distribution = DiscreteScoreDistribution().fit(calibration, history, prediction_start=forecast.gameday.min())
            fold = errors.loc[errors.model.eq(name) & errors.season.eq(season)]
            aligned = forecast.merge(fold[["game_id", "gameday", "mean_fit_end", "pred_home_score", "pred_away_score"]], on=["game_id", "gameday"], how="left", validate="one_to_one")
            if aligned[["pred_home_score", "pred_away_score", "mean_fit_end"]].isna().any().any():
                raise ValueError(f"Missing OOT means for {name}/{season}")
            if not aligned.mean_fit_end.lt(forecast.gameday.min()).all():
                raise ValueError("Seasonal means must be fitted before the outer season starts")
            if progress:
                progress(f"{season} {name}: {len(calibration)} residual pairs, {len(history)} historical score pairs; price {len(forecast)} games")
            fit_logs.append({"season": season, "model": name, "calibration_games": len(calibration), "score_history_games": len(history),
                             "calibration_end": distribution.residual_model_.calibration_end_, "score_history_end": distribution.score_history_end_,
                             "tie_history_games": distribution.regular_history_games_, "tie_history_count": distribution.regular_history_ties_,
                             "tie_prior_rate": distribution.tie_prior_rate_, "tie_odds_scale": distribution.tie_scale_,
                             "max_score": distribution.max_score, "bandwidth": distribution.bandwidth})
            for row in aligned.itertuples(index=False):
                grid = distribution.predict_grid([row.pred_home_score, row.pred_away_score], season=row.season, game_type=row.game_type)
                season_rows.extend(price_game(row, grid, model=name, phase=phase, rules=rules))
            if season == audit_season and mean_models is not None:
                mean_model = mean_models[name].mean_model
                recomputed = mean_model.predict(forecast)[["home_score", "away_score"]].to_numpy()
                if not np.allclose(recomputed, aligned[["pred_home_score", "pred_away_score"]].to_numpy(), atol=1e-8, rtol=1e-8):
                    raise ValueError("Saved mean model disagrees with cached OOT forecasts")
                cutoff = pd.Timestamp(mean_models[name].mean_fit_end)
                if cutoff >= forecast.gameday.min():
                    raise ValueError("Saved mean model overlaps the audit season")
                final_models[name] = DiscreteGameModel(mean_model, distribution, cutoff)
        # Book prices imply conditional two-sided probabilities, not push mass.
        for row in season_rows:
            exhaustive_pair = row["market"] != "moneyline" or rules.moneyline_tie == "push"
            if row["model"] == MODEL_NAMES[0] and exhaustive_pair and pd.notna(row["odds"]) and pd.notna(row["opposite_odds"]):
                benchmark = {**row, "model": "no_vig_prices", "p_win": np.nan, "p_loss": np.nan, "p_push": np.nan,
                             "p_conditional": no_vig_probability(row["odds"], row["opposite_odds"])}
                for column in ("expected_net_profit_per_unit", "edge_vs_price_conditional", "model_probability_conditional"):
                    benchmark[column] = np.nan
                predictions.append(benchmark)
        predictions.extend(season_rows)
    frame = pd.DataFrame(predictions)
    metrics, bins = summarize_probabilities(frame)
    metadata = {
        "schema_version": 1, "first_evaluation_season": first_evaluation_season, "audit_season": audit_season,
        "moneyline_tie": rules.moneyline_tie, "includes_overtime": rules.includes_overtime,
        "scoring_model": "Integer final-score grid; correlated Gaussian OOT residual density times learned local score/margin frequency corrections. Tie odds calibrated on earlier REG outcomes; no postseason ties.",
        "grid_policy": "Scores 0..100 excluding 1, using common 2/3/6/7/8-point scoring increments. Upper Gaussian tail bound must be <= 1e-6. No rounding of continuous samples.",
        "fixed_parameters": {"bandwidth": 2.0, "count_pseudocount": 1.0, "correction_bounds": [.25, 4.0], "tie_beta_prior": [.5, .5]},
        "evaluation": "Canonical sides only: home moneyline, home spread, over total. Conditional binary metrics exclude actual pushes; three-class scores and push diagnostics include them. Exact grid probabilities, not Monte Carlo frequency estimates.",
        "limitations": [
            "2025 is a previously inspected retrospective audit; no settings are selected against these results.",
            "Final scores already include overtime. OT possessions and regulation scores are not separately simulated; rule versions describe terminal constraints.",
            "2025 regular-season overtime rules changed. Earlier tie-rate history is pooled across regimes; there is no pre-2025 calibration sample under the new REG rules.",
            "Rare defensive one-point safety finals are outside this model's scoring support; history containing such a score is rejected.",
            "Score-grid shape and regularization are fixed research assumptions; probability calibration must be assessed, not presumed.",
            "Settlement is an explicit full-game contract, not a reconstruction of each historical sportsbook's house rules. Regulation-only, quarter-line, voided and abandoned markets are unsupported.",
            "If moneyline ties lose, no two-way no-vig moneyline benchmark is calculated: the draw outcome would require a third price.",
            "Prices are untimestamped historical snapshots. Expected values are diagnostic; no executable betting edge, strategy ROI, or betting recommendation is claimed.",
        ],
        "rule_sources": ["https://operations.nfl.com/media/ntif5hxb/2025-nfl-rulebook-final.pdf", "https://operations.nfl.com/media/24emxacq/2024-nfl-rulebook.pdf", "https://operations.nfl.com/media/nppjkdp1/2023-record-and-fact-book.pdf", "https://sportsbook.draftkings.com/help/general-betting-rules/general-rules"],
    }
    return MarketProbabilityRun(frame, metrics, bins, pd.DataFrame(fit_logs), final_models, metadata)
