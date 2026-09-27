"""Rolling-origin opportunity benchmarks with saved fitting boundaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_model.distribution_backtesting import inner_temporal_split
from nfl_model.opportunity_models import OPPORTUNITIES, VOLUME_TARGETS, PlayerOpportunityModel, TeamVolumeModel


@dataclass
class OpportunityRun:
    volume_predictions: pd.DataFrame
    player_predictions: pd.DataFrame
    volume_metrics: pd.DataFrame
    player_metrics: pd.DataFrame
    participation_metrics: pd.DataFrame
    tuning: pd.DataFrame
    fold_models: dict


def _metrics(frame, groups):
    records = []
    for keys, rows in frame.groupby(groups, sort=False):
        error = rows.predicted - rows.actual
        records.append({**dict(zip(groups, keys)), "rows": len(rows), "mae": float(error.abs().mean()),
                        "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean())})
    return pd.DataFrame(records)


def run_opportunity_backtest(games: pd.DataFrame, players: pd.DataFrame, *,
                             alphas=(100., 1000., 10000.), progress=None) -> OpportunityRun:
    volume_rows, player_rows, activity_rows, tuning_rows, models = [], [], [], [], {}
    for season in range(2022, 2026):
        train, forecast = games.loc[games.season.lt(season)], games.loc[games.season.eq(season)]
        train_players, forecast_players = players.loc[players.season.lt(season)], players.loc[players.season.eq(season)]
        if train.empty or forecast.empty or train.gameday.max() >= forecast.gameday.min():
            raise ValueError("Opportunity folds must be complete and chronological")
        if progress:
            progress(f"{season}: select team-volume regularization; fit player participation and shares on {len(train_players):,} earlier rows")
        inner_train, inner_valid = inner_temporal_split(train)
        candidates = []
        for alpha in alphas:
            candidate = TeamVolumeModel(alpha=alpha).fit(inner_train)
            error = candidate.predict(inner_valid).to_numpy() - inner_valid[list(VOLUME_TARGETS)].to_numpy()
            rmse = float(np.sqrt(np.square(error).mean()))
            candidates.append((rmse, -alpha, alpha))
            tuning_rows.append({"season": season, "alpha": alpha, "rmse": rmse, "inner_train_end": inner_train.gameday.max(),
                                "inner_valid_start": inner_valid.gameday.min(), "inner_valid_end": inner_valid.gameday.max()})
        chosen = min(candidates)[2]
        volumes = {name: TeamVolumeModel(kind=name, alpha=chosen).fit(train) for name in ("ridge", "recent_average", "training_mean")}
        allocation = {name: PlayerOpportunityModel(kind=name).fit(train_players) for name in ("trained", "recent_average")}
        target_rate = float(train_players.groupby(["game_id", "team"]).team_targets.first().sum() / train_players.groupby(["game_id", "team"]).team_attempts.first().sum())
        if not 0 <= target_rate <= 1:
            raise ValueError("Team targets cannot exceed pass attempts")
        models[season] = {"volume": volumes["ridge"], "players": allocation["trained"], "target_rate": target_rate,
                          "fit_end": train.gameday.max(), "selected_alpha": chosen}
        team_forecasts = None
        for name, model in volumes.items():
            predicted = model.predict(forecast)
            for target in VOLUME_TARGETS:
                for i, row in forecast.iterrows():
                    volume_rows.append({"game_id": row.game_id, "season": season, "gameday": row.gameday, "model": name,
                                        "target": target, "actual": row[target], "predicted": predicted.loc[i, target], "fit_end": train.gameday.max()})
            if name == "ridge":
                blocks = []
                for side in ("home", "away"):
                    part = forecast[["game_id", f"{side}_team"]].rename(columns={f"{side}_team": "team"}).copy()
                    for target in ("attempts", "carries"):
                        part[target] = predicted[f"{side}_{target}"].to_numpy()
                    part["targets"] = part.attempts * target_rate
                    blocks.append(part)
                team_forecasts = pd.concat(blocks, ignore_index=True).set_index(["game_id", "team"])
        for name, model in allocation.items():
            weights = model.predict_weights(forecast_players)
            shares = model.predict_shares(forecast_players)
            team = team_forecasts.reindex(pd.MultiIndex.from_frame(forecast_players[["game_id", "team"]]))
            for target in OPPORTUNITIES:
                predicted = shares[target].to_numpy() * team[target].to_numpy()
                for j, row in enumerate(forecast_players.itertuples()):
                    player_rows.append({"game_id": row.game_id, "team": row.team, "player_id": row.player_id, "season": season,
                                        "model": name, "target": target, "position": row.position, "is_other": row.is_other,
                                        "established": getattr(row, f"prior_{target}_last5") > 0,
                                        "actual": getattr(row, target), "predicted": predicted[j], "predicted_share": shares[target].iloc[j],
                                        "actual_share": getattr(row, target) / getattr(row, f"team_{target}") if getattr(row, f"team_{target}") else np.nan,
                                        "fit_end": train.gameday.max()})
            known = forecast_players.offense_active.notna() & forecast_players.is_other.eq(0)
            truth = forecast_players.loc[known, "offense_active"].to_numpy()
            probabilities = weights.loc[known, "active_probability"].to_numpy()
            clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
            activity_rows.append({"season": season, "model": name, "rows": int(known.sum()), "brier": float(np.square(probabilities - truth).mean()),
                                  "log_loss": float(-(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)).mean()),
                                  "predicted_active": float(probabilities.mean()), "actual_active": float(truth.mean())})
    volume_frame, player_frame = pd.DataFrame(volume_rows), pd.DataFrame(player_rows)
    volume_metrics = _metrics(volume_frame, ["season", "model", "target"])
    player_metrics = []
    for label, subset in (("all_named", player_frame.loc[player_frame.is_other.eq(0)]),
                           ("prior_usage", player_frame.loc[player_frame.is_other.eq(0) & player_frame.established]),
                           ("other", player_frame.loc[player_frame.is_other.eq(1)])):
        player_metrics.append(_metrics(subset, ["season", "model", "target"]).assign(cohort=label))
    return OpportunityRun(volume_frame, player_frame, volume_metrics, pd.concat(player_metrics, ignore_index=True), pd.DataFrame(activity_rows), pd.DataFrame(tuning_rows), models)
