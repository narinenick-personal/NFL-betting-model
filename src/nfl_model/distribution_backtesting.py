"""Nested chronological means and prequential residual-distribution evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd

from nfl_model.backtesting import BacktestConfig, DEFAULT_ALPHAS, chronological_splits, select_ridge_alpha
from nfl_model.distributions import DISTRIBUTIONS, JointResidualDistribution, JointScoreModel, game_rng
from nfl_model.distribution_evaluation import evaluate_draws, summarize_distributions
from nfl_model.models import MarketLineModel, RidgeScoreModel
from nfl_model.team_games import TARGET_COLUMNS

MODEL_NAMES = ("ridge_football", "ridge_market", "market_lines")


@dataclass(frozen=True)
class DistributionConfig:
    first_season: int = 2021
    audit_season: int = 2025
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    draws: int = 10_000
    seed: int = 20260925
    min_calibration_games: int = 100

    def validate(self) -> None:
        if self.audit_season - self.first_season < 3:
            raise ValueError("Need at least four seasons: initial training, residual warmup, development, audit")
        if isinstance(self.draws, bool) or not isinstance(self.draws, int) or self.draws < 2:
            raise ValueError("draws must be an integer >= 2")
        if self.min_calibration_games < 3:
            raise ValueError("Need at least three calibration games")
        BacktestConfig(tuple(range(self.first_season, self.audit_season - 1)), self.audit_season - 1, self.audit_season, self.alphas).validate()


def inner_temporal_split(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use latest prior season for validation, or last quarter of the first season."""
    last = history.season.max()
    if history.season.nunique() > 1:
        train, validation = history.loc[history.season.lt(last)], history.loc[history.season.eq(last)]
    else:
        dates = sorted(pd.to_datetime(history.gameday).unique())
        if len(dates) < 2:
            raise ValueError("Initial season needs at least two distinct game dates")
        cutoff = dates[min(max(int(len(dates) * .75), 1), len(dates) - 1)]
        train, validation = history.loc[history.gameday.lt(cutoff)], history.loc[history.gameday.ge(cutoff)]
    if train.empty or validation.empty or not train.gameday.max() < validation.gameday.min():
        raise ValueError("Inner tuning split is not strictly chronological")
    return train.copy(), validation.copy()


def _fit_outer_means(history: pd.DataFrame, season: int, alphas: tuple[float, ...]) -> tuple[dict, pd.DataFrame]:
    inner_train, inner_validation = inner_temporal_split(history)
    models = {"market_lines": MarketLineModel()}
    tables = []
    for name in MODEL_NAMES[:2]:
        chosen, table = select_ridge_alpha(inner_train, inner_validation, include_market=name == "ridge_market", alphas=alphas)
        models[name] = RidgeScoreModel(alpha=chosen.alpha, include_market=name == "ridge_market").fit(history)
        table = table.assign(model=name, forecast_season=season,
                             inner_train_end=inner_train.gameday.max(), inner_validation_start=inner_validation.gameday.min(),
                             inner_validation_end=inner_validation.gameday.max(), mean_fit_end=history.gameday.max())
        tables.append(table)
    return models, pd.concat(tables, ignore_index=True)


def select_distributions(development_joints: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    """Select distribution family by average joint energy; no audit input accepted."""
    if development_joints.empty or not development_joints.phase.eq("development").all():
        raise ValueError("Distribution selection accepts development rows only")
    selection = development_joints.groupby(["model", "distribution"], sort=False).agg(
        energy_score=("energy_score", "mean"), games=("game_id", "size")
    ).reset_index()
    chosen = {}
    for model, group in selection.groupby("model"):
        if set(group.distribution) != set(DISTRIBUTIONS):
            raise ValueError("Both distribution candidates must be evaluated")
        ids = development_joints.loc[development_joints.model.eq(model)].groupby("distribution").game_id.apply(set)
        if ids.iloc[0] != ids.iloc[1]:
            raise ValueError("Distribution candidates need identical development games")
        chosen[model] = group.sort_values(["energy_score", "distribution"]).iloc[0].distribution
    selection["selected"] = selection.apply(lambda row: chosen[row.model] == row.distribution, axis=1)
    return chosen, selection


@dataclass
class DistributionRun:
    residuals: pd.DataFrame
    tuning: pd.DataFrame
    diagnostics: pd.DataFrame
    joint_diagnostics: pd.DataFrame
    metrics: pd.DataFrame
    joint_metrics: pd.DataFrame
    pit_histograms: pd.DataFrame
    selection: pd.DataFrame
    models: dict[str, JointScoreModel]
    metadata: dict


def run_distribution_backtest(
    games: pd.DataFrame, config: DistributionConfig | None = None, *, progress: Callable[[str], None] | None = None
) -> DistributionRun:
    config = config or DistributionConfig()
    config.validate()
    splits = chronological_splits(games, BacktestConfig(tuple(range(config.first_season, config.audit_season - 1)), config.audit_season - 1, config.audit_season, config.alphas))
    data = pd.concat(splits.values(), ignore_index=True).sort_values(["gameday", "game_id"])
    residual_frames, tuning_frames, marginal_rows, joint_rows = [], [], [], []
    selected, selection, final_models = {}, pd.DataFrame(), {}
    for season in range(config.first_season + 1, config.audit_season + 1):
        if progress:
            progress(f"Season {season}: tune and fit means using only earlier games...")
        history, forecast = data.loc[data.season.lt(season)], data.loc[data.season.eq(season)]
        if history.empty or forecast.empty or not history.gameday.max() < forecast.gameday.min():
            raise ValueError(f"Incomplete or overlapping outer fold for {season}")
        means, tuning = _fit_outer_means(history, season, config.alphas)
        tuning_frames.append(tuning)
        phase = "retrospective_audit" if season == config.audit_season else "development"
        if season == config.audit_season:
            # Freeze family choice before seeing any audit outcomes or diagnostics.
            selected, selection = select_distributions(pd.DataFrame(joint_rows))
        for name in MODEL_NAMES:
            predictions = means[name].predict(forecast)
            if not np.isfinite(predictions.to_numpy(dtype=float)).all():
                raise ValueError(f"{name} has missing projections; matched distribution comparisons require complete score means")
            fold = forecast[["game_id", "season", "gameday", *TARGET_COLUMNS]].copy()
            fold["model"] = name
            fold["phase"] = "residual_warmup" if season == config.first_season + 1 else phase
            fold["calibration_eligible"] = season < config.audit_season
            fold["mean_fit_end"] = history.gameday.max()
            for target in TARGET_COLUMNS:
                fold[f"pred_{target}"] = predictions[target]
                fold[f"residual_{target}"] = fold[target] - predictions[target]
            if season > config.first_season + 1:
                past = pd.concat(residual_frames, ignore_index=True)
                calibration = past.loc[past.model.eq(name) & past.season.lt(season)]
                if len(calibration) < config.min_calibration_games:
                    raise ValueError(f"Insufficient prior residuals for {name}/{season}: {len(calibration)}")
                if progress:
                    progress(f"  {name}: {len(calibration)} prior residual pairs; {len(forecast)} games × {config.draws:,} draws per candidate")
                for kind in DISTRIBUTIONS:
                    distribution = JointResidualDistribution(kind).fit(calibration, prediction_start=forecast.gameday.min())
                    if season == config.audit_season and selected[name] == kind:
                        final_models[name] = JointScoreModel(means[name], distribution, history.gameday.max())
                    for row in fold.itertuples(index=False):
                        samples = distribution.sample([row.pred_home_score, row.pred_away_score], draws=config.draws, rng=game_rng(config.seed, row.game_id, kind))
                        marginal, joint = evaluate_draws([getattr(row, target) for target in TARGET_COLUMNS], samples, pit_rng=game_rng(config.seed, row.game_id, "pit"))
                        identity = {"game_id": row.game_id, "season": season, "phase": phase, "model": name, "distribution": kind,
                                    "calibration_games": len(calibration), "calibration_end": distribution.calibration_end_, "mean_fit_end": history.gameday.max()}
                        marginal_rows.extend({**identity, **item} for item in marginal)
                        joint_rows.append({**identity, **joint})
            # Only after forecasting the entire season do its residuals join history.
            residual_frames.append(fold)
    diagnostics, joints = pd.DataFrame(marginal_rows), pd.DataFrame(joint_rows)
    metrics, joint_metrics, pits = summarize_distributions(diagnostics, joints)
    for frame in (diagnostics, joints, metrics, joint_metrics, pits):
        frame["selected"] = frame.apply(lambda row: selected[row.model] == row.distribution, axis=1)
    metadata = {
        "schema_version": 1, "first_season": config.first_season, "audit_season": config.audit_season,
        "residual_seasons": list(range(config.first_season + 1, config.audit_season)),
        "development_seasons": list(range(config.first_season + 2, config.audit_season)),
        "draws_per_game": config.draws, "seed": config.seed, "alpha_grid": list(config.alphas),
        "selected_distributions": selected, "selection_metric": "Mean joint energy score on prequential development forecasts only",
        "final_calibration_rows": {name: model.distribution.calibration_rows_ for name, model in final_models.items()},
        "final_mean_alphas": {name: model.mean_model.alpha for name, model in final_models.items() if isinstance(model.mean_model, RidgeScoreModel)},
        "protocol": "Nested chronological mean tuning; latest prior season validates alpha, except first outer fold uses a date-grouped 75/25 split of the initial season. Refit on all prior seasons. Each distribution uses earlier out-of-time paired residuals only.",
        "sample_policy": "10,000 draws by default from fitted residual families around trained score means, retaining estimated error bias and within-game dependence, with scores censored at zero. No integer rounding.",
        "limitations": [
            "The audit season was inspected during baseline work; this is a retrospective audit, not a fresh untouched holdout.",
            "Continuous score distributions do not represent NFL scoring increments, tie/overtime rules, or push probabilities; no sportsbook probabilities or EV are emitted.",
            "Residual scale is constant across matchups within a fold; game-dependent volatility and player opportunity are not modeled.",
            "Coverage Wilson intervals are descriptive binomial intervals; shared-team/game dependence and multiple comparisons limit formal interpretations.",
            "Model families and hyperparameter grids were designed retrospectively; nested fits enforce data chronology but cannot make this a prospective study.",
            "Current nflverse snapshots and untimestamped market lines retain the upstream availability limitations documented in the dataset manifest.",
        ],
    }
    return DistributionRun(pd.concat(residual_frames, ignore_index=True), pd.concat(tuning_frames, ignore_index=True), diagnostics, joints, metrics, joint_metrics, pits, selection, final_models, metadata)
