"""Chronological player production fits, simulations, and diagnostic thresholds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_model.discrete_scores import DiscreteGameModel, DiscreteScoreDistribution
from nfl_model.distribution_evaluation import ensemble_crps
from nfl_model.distributions import game_rng
from nfl_model.player_dataset import PLAYER_TARGETS
from nfl_model.player_production import PlayerProductionModel, TouchdownModel, production_calibration, fit_allocation_concentrations
from nfl_model.player_simulation import ConditionalVolumeDistribution, PlayerGameSimulator

# Fixed research thresholds, NOT historical sportsbook lines.
DIAGNOSTICS = {
    "attempts": ("attempts", 24.5), "carries": ("carries", 9.5), "targets": ("targets", 4.5),
    "receptions": ("targets", 3.5), "passing_yards": ("attempts", 199.5),
    "rushing_yards": ("carries", 39.5), "receiving_yards": ("targets", 39.5),
    "passing_tds": ("attempts", .5), "rushing_tds": ("carries", .5), "receiving_tds": ("targets", .5),
}


@dataclass
class CachedScoreMeans:
    predictions: pd.DataFrame

    def predict(self, games):
        values = self.predictions.set_index("game_id").reindex(games.game_id)
        result = pd.DataFrame({"home_score": values.pred_home_score.to_numpy(), "away_score": values.pred_away_score.to_numpy()}, index=games.index)
        if not np.isfinite(result).all().all():
            raise ValueError("Missing cached score means")
        return result


def _validate_draws(scores, records, game_row):
    for side in ("home", "away"):
        record = records[getattr(game_row, f"{side}_team")]
        p, volume = record["samples"], record["volume"]
        for stat in ("attempts", "carries"):
            if not np.array_equal(p[stat].sum(axis=1), volume[stat]):
                raise ValueError("Player/team opportunity reconciliation failed")
        identities = (("completions", "receptions"), ("passing_yards", "receiving_yards"), ("passing_tds", "receiving_tds"))
        for left, right in identities:
            if not np.array_equal(p[left].sum(axis=1), p[right].sum(axis=1)):
                raise ValueError(f"{left}/{right} reconciliation failed")
        for count, cap in (("receptions", "targets"), ("completions", "attempts"), ("receiving_tds", "receptions"), ("passing_tds", "completions"), ("rushing_tds", "carries")):
            if (p[count] > p[cap]).any():
                raise ValueError(f"{count} exceeds player capacity")
        if (p["targets"].sum(axis=1) > volume["attempts"]).any():
            raise ValueError("Targets exceed attempts")
        if (6 * (p["passing_tds"] + p["rushing_tds"]).sum(axis=1) > scores[f"{side}_score"]).any():
            raise ValueError("Touchdowns exceed team score")
        for stat in PLAYER_TARGETS:
            if (p[stat][p["offense_active"] == 0] != 0).any():
                raise ValueError("Inactive player received production")


def _baseline_rates(history):
    result = {}
    for stat, (usage, threshold) in DIAGNOSTICS.items():
        eligible = history.loc[history.is_other.eq(0) & history[f"prior_{usage}_last5"].gt(0)]
        grouped = eligible.assign(over=eligible[stat].gt(threshold)).groupby("position").agg(over=("over", "sum"), rows=("over", "size"), mean=(stat, "mean"))
        result[stat] = {p: {"probability": (r.over + .5) / (r.rows + 1), "mean": r["mean"]} for p, r in grouped.iterrows()}
    return result


def summarize_player_simulations(frame):
    rows = []
    for (season, stat), part in frame.groupby(["season", "statistic"], sort=False):
        for model, prediction, probability in (("simulation", "mean", "p_over"), ("recent_average", "recent_average", None), ("position_frequency", None, "baseline_p_over")):
            record = {"season": season, "statistic": stat, "model": model, "rows": len(part)}
            if prediction:
                error = part[prediction] - part.actual
                record.update(mae=float(error.abs().mean()), rmse=float(np.sqrt(np.square(error).mean())), bias=float(error.mean()))
            if probability:
                p = np.clip(part[probability], 1e-15, 1 - 1e-15)
                y = part.actual.gt(part.threshold).astype(float)
                record.update(brier=float(np.square(part[probability] - y).mean()), log_loss=float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()))
            if model == "simulation":
                record.update(crps=float(part.crps.mean()), coverage_80=float(part.covered_80.mean()),
                              interval_mass_80=float(part.interval_mass_80.mean()), width_80=float((part.upper_80 - part.lower_80).mean()))
            rows.append(record)
    return pd.DataFrame(rows)


@dataclass
class PlayerSimulationRun:
    diagnostics: pd.DataFrame
    metrics: pd.DataFrame
    production_predictions: pd.DataFrame
    fit_log: pd.DataFrame
    example_draws: pd.DataFrame
    models: dict


def run_player_simulation_backtest(games, players, teams, score_residuals, volume_predictions, opportunity_models,
                                   *, final_score_model, draws=10_000, seed=20260925, progress=None):
    if not pd.to_datetime(score_residuals.mean_fit_end).lt(pd.to_datetime(score_residuals.gameday)).all():
        raise ValueError("Score forecasts must be out of time")
    production_history, allocation_history, diagnostic_rows, fit_rows, examples, final_models = [], [], [], [], [], {}
    for season in range(2022, 2026):
        train = players.loc[players.season.lt(season)]
        forecast = players.loc[players.season.eq(season)]
        game_history = games.loc[games.season.lt(season)]
        forecast_games = games.loc[games.season.eq(season)]
        if train.empty or forecast.empty or train.gameday.max() >= forecast.gameday.min():
            raise ValueError("Invalid chronological production fold")
        bundle = opportunity_models[season]
        if pd.Timestamp(bundle["fit_end"]) >= forecast.gameday.min():
            raise ValueError("Opportunity bundle overlaps forecast period")
        if progress:
            progress(f"{season}: fit production efficiencies on {len(train):,} prior rows")
        production = PlayerProductionModel().fit(train)
        predicted = production.predict(forecast)
        fold = forecast[["game_id", "team", "player_id", "season", "gameday", "is_other", *PLAYER_TARGETS]].copy()
        for col in predicted:
            fold[col] = predicted[col].to_numpy()
        fold["fit_end"] = train.gameday.max()
        allocation = forecast[["game_id", "team", "player_id", "season", "gameday", "is_other", "offense_active", "attempts", "carries", "targets"]].copy()
        weights = bundle["players"].predict_weights(forecast)
        for col in weights:
            allocation[col] = weights[col].to_numpy()
        allocation["fit_end"] = train.gameday.max()
        if season > 2022:
            parameters = production_calibration(pd.concat(production_history, ignore_index=True))
            allocations = fit_allocation_concentrations(pd.concat(allocation_history, ignore_index=True))
            earlier_scores = score_residuals.loc[score_residuals.season.lt(season)]
            earlier_volume = volume_predictions.loc[volume_predictions.season.lt(season)]
            volume_distribution = ConditionalVolumeDistribution().fit(earlier_scores, earlier_volume, prediction_start=forecast.gameday.min())
            discrete = DiscreteScoreDistribution().fit(earlier_scores, game_history, prediction_start=forecast.gameday.min())
            cached = score_residuals.loc[score_residuals.season.eq(season)]
            if not pd.to_datetime(cached.mean_fit_end).lt(forecast_games.gameday.min()).all():
                raise ValueError("Cached means overlap their forecast season")
            score_model = DiscreteGameModel(CachedScoreMeans(cached), discrete, game_history.gameday.max())
            if season == 2025:
                actual_means = final_score_model.mean_model.predict(forecast_games)[["home_score", "away_score"]]
                expected_means = CachedScoreMeans(cached).predict(forecast_games)
                if not np.allclose(actual_means, expected_means, atol=1e-8):
                    raise ValueError("Final score model disagrees with cached forecasts")
                score_model = final_score_model
            touchdown = TouchdownModel().fit(teams.loc[teams.season.lt(season)])
            simulator = PlayerGameSimulator(score_model, bundle["volume"], bundle["players"], production, volume_distribution,
                                             bundle["target_rate"], allocations, parameters, touchdown, train.gameday.max())
            if season == 2025:
                final_models["football_players"] = simulator
            fit_rows.append({"season": season, "fit_end": train.gameday.max(), "production_calibration_end": parameters["calibration_end"],
                             "volume_calibration_end": volume_distribution.calibration_end_, "calibration_games": len(volume_distribution.calibration_ids_),
                             "catch_rho": parameters["catch_intraclass_correlation"],
                             **{f"{name}_concentration": value["concentration"] for name, value in allocations.items()}})
            baselines = _baseline_rates(train)
            game_lookup = {r.game_id: r for r in forecast_games.itertuples()}
            for number, (game_id, scores, volume, records) in enumerate(simulator.simulate(forecast_games, forecast, draws=draws, seed=seed)):
                _validate_draws(scores, records, game_lookup[game_id])
                save_example = season == 2025 and not examples and any(r["candidates"].is_other.eq(0).any() for r in records.values())
                for team, record in records.items():
                    candidates, arrays = record["candidates"], record["samples"]
                    if save_example:
                        for i, candidate in enumerate(candidates.itertuples()):
                            examples.append(pd.DataFrame({"game_id": game_id, "team": team, "player_id": candidate.player_id,
                                                          "draw_id": np.arange(draws), **{name: array[:, i] for name, array in arrays.items()}}))
                    for i, candidate in enumerate(candidates.itertuples()):
                        if candidate.is_other:
                            continue
                        for stat, (usage, threshold) in DIAGNOSTICS.items():
                            if not getattr(candidate, f"prior_{usage}_last5") > 0:
                                continue
                            values = arrays[stat][:, i]
                            lower, upper = np.quantile(values, [.1, .9])
                            actual = getattr(candidate, stat)
                            baseline = baselines[stat].get(candidate.position, {"probability": .5, "mean": 0})
                            recent = getattr(candidate, f"prior_{stat}_last5")
                            pit_rng = game_rng(seed, game_id, f"player_pit_{team}_{candidate.player_id}_{stat}")
                            pit = float((values < actual).mean() + pit_rng.random() * (values == actual).mean())
                            diagnostic_rows.append({"season": season, "game_id": game_id, "team": team, "player_id": candidate.player_id,
                                                    "position": candidate.position, "statistic": stat, "threshold": threshold, "actual": actual,
                                                    "mean": float(values.mean()), "lower_80": lower, "upper_80": upper,
                                                    "covered_80": lower <= actual <= upper, "crps": ensemble_crps(actual, values),
                                                    "interval_mass_80": float(((values >= lower) & (values <= upper)).mean()), "pit": pit,
                                                    "p_over": float((values > threshold).mean()), "baseline_p_over": baseline["probability"],
                                                    "recent_average": recent if pd.notna(recent) else baseline["mean"],
                                                    "p_offense_active": float(arrays["offense_active"][:, i].mean()), "observed_offense_active": candidate.offense_active})
                if progress and (number + 1) % 25 == 0:
                    progress(f"{season}: simulated and checked {number + 1}/{len(forecast_games)} games × {draws:,} draws")
        # Add this season's outcomes only after its forecasts are complete.
        production_history.append(fold)
        allocation_history.append(allocation)
    diagnostics = pd.DataFrame(diagnostic_rows)
    return PlayerSimulationRun(diagnostics, summarize_player_simulations(diagnostics), pd.concat(production_history, ignore_index=True),
                                pd.DataFrame(fit_rows), pd.concat(examples, ignore_index=True), final_models)
