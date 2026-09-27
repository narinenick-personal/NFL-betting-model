"""Chronological baseline experiment: validation selects alpha, test only scores it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_model.data_quality import require_columns, require_unique_keys
from nfl_model.evaluation import evaluation_table, regression_metrics
from nfl_model.modeling_dataset import feature_groups
from nfl_model.models import MarketLineModel, RecentTeamAverageModel, RidgeScoreModel, SCORE_TARGETS, TrainingMeanModel
from nfl_model.team_games import TARGET_COLUMNS

DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


@dataclass(frozen=True)
class BacktestConfig:
    train_seasons: tuple[int, ...] = (2021, 2022, 2023)
    validation_season: int = 2024
    test_season: int = 2025
    alphas: tuple[float, ...] = DEFAULT_ALPHAS

    def validate(self) -> None:
        if not self.train_seasons or len(set(self.train_seasons)) != len(self.train_seasons):
            raise ValueError("Training seasons must be nonempty and distinct")
        if not max(self.train_seasons) < self.validation_season < self.test_season:
            raise ValueError("Require training seasons < validation season < test season")
        if not self.alphas or len(set(self.alphas)) != len(self.alphas) or any(not np.isfinite(a) or a <= 0 for a in self.alphas):
            raise ValueError("Alphas must be distinct, finite and positive")


def chronological_splits(games: pd.DataFrame, config: BacktestConfig) -> dict[str, pd.DataFrame]:
    config.validate()
    require_unique_keys(games, ["game_id"], "modeling dataset")
    require_columns(games, ["season", "gameday", *TARGET_COLUMNS], "modeling dataset")
    selected_seasons = {*config.train_seasons, config.validation_season, config.test_season}
    missing = selected_seasons - set(games.season.unique())
    if missing:
        raise ValueError(f"Dataset is missing requested seasons: {sorted(missing)}")
    selected = games.loc[games.season.isin(selected_seasons)].copy()
    selected["gameday"] = pd.to_datetime(selected.gameday, errors="raise")
    if selected.gameday.isna().any():
        raise ValueError("Game dates cannot be missing")
    targets = selected[list(TARGET_COLUMNS)].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(targets).all() or (selected[list(SCORE_TARGETS)] < 0).any().any():
        raise ValueError("Targets must be finite with nonnegative team scores")
    if not np.allclose(selected.game_total, selected.home_score + selected.away_score) or not np.allclose(selected.home_margin, selected.home_score - selected.away_score):
        raise ValueError("Targets must reconcile with team scores")
    selected = selected.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    splits = {
        "train": selected.loc[selected.season.isin(config.train_seasons)].copy(),
        "validation": selected.loc[selected.season.eq(config.validation_season)].copy(),
        "test": selected.loc[selected.season.eq(config.test_season)].copy(),
    }
    if not splits["train"].gameday.max() < splits["validation"].gameday.min() or not splits["validation"].gameday.max() < splits["test"].gameday.min():
        raise ValueError("Game dates overlap between chronological splits")
    return splits


def select_ridge_alpha(
    train: pd.DataFrame, validation: pd.DataFrame, *, include_market: bool, alphas: tuple[float, ...]
) -> tuple[RidgeScoreModel, pd.DataFrame]:
    """Select by pooled home/away score RMSE; this function has no test input."""
    if not alphas or len(set(alphas)) != len(alphas) or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Alphas must be distinct, finite and positive")
    candidates = []
    fitted = {}
    for alpha in sorted(alphas):
        model = RidgeScoreModel(alpha=alpha, include_market=include_market).fit(train)
        predictions = model.predict(validation)
        score_metrics = regression_metrics(validation[list(SCORE_TARGETS)], predictions[list(SCORE_TARGETS)])
        candidates.append({
            "alpha": alpha, "score_rmse": score_metrics["rmse"], "score_mae": score_metrics["mae"],
            "total_rmse": regression_metrics(validation.game_total, predictions.game_total)["rmse"],
            "margin_rmse": regression_metrics(validation.home_margin, predictions.home_margin)["rmse"],
        })
        fitted[alpha] = model
    table = pd.DataFrame(candidates)
    # Deterministic tie break: prefer more regularization.
    best = table.sort_values(["score_rmse", "alpha"], ascending=[True, False]).iloc[0]
    table["selected"] = table.alpha.eq(best.alpha)
    return fitted[best.alpha], table


@dataclass
class BaselineRun:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    tuning: pd.DataFrame
    models: dict
    metadata: dict


def run_baseline_backtest(games: pd.DataFrame, config: BacktestConfig | None = None) -> BaselineRun:
    config = config or BacktestConfig()
    splits = chronological_splits(games, config)
    train, validation, test = (splits[key] for key in ("train", "validation", "test"))
    validation_models = {
        "training_mean": TrainingMeanModel().fit(train),
        "recent_team_average": RecentTeamAverageModel().fit(train),
        "market_lines": MarketLineModel(),
    }
    tuning_tables = []
    selected_alphas = {}
    for name, use_market in (("ridge_football", False), ("ridge_market", True)):
        model, tuning = select_ridge_alpha(train, validation, include_market=use_market, alphas=config.alphas)
        validation_models[name] = model
        selected_alphas[name] = float(model.alpha)
        tuning.insert(0, "model", name)
        tuning_tables.append(tuning)

    # All choices are frozen before the test season is used for any evaluation.
    refit = pd.concat([train, validation], ignore_index=True)
    final_models = {
        "training_mean": TrainingMeanModel().fit(refit),
        "recent_team_average": RecentTeamAverageModel().fit(refit),
        "market_lines": MarketLineModel(),
        **{name: RidgeScoreModel(alpha=alpha, include_market=(name == "ridge_market")).fit(refit) for name, alpha in selected_alphas.items()},
    }
    prediction_frames = []
    for split, frame, models, fit_through in (
        ("validation", validation, validation_models, max(config.train_seasons)),
        ("test", test, final_models, config.validation_season),
    ):
        for name, model in models.items():
            prediction = model.predict(frame)
            rows = frame[["game_id", "season", "week", "gameday", "home_team", "away_team", "game_type", *TARGET_COLUMNS]].copy()
            rows["split"] = split
            rows["model"] = name
            rows["fit_through_season"] = fit_through if name != "market_lines" else pd.NA
            for target in TARGET_COLUMNS:
                rows[f"pred_{target}"] = prediction[target]
                rows[f"residual_{target}"] = rows[target] - prediction[target]
            prediction_frames.append(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["fit_through_season"] = predictions.fit_through_season.astype("Int64")
    metadata = {
        "schema_version": 1,
        "train_seasons": sorted(config.train_seasons),
        "validation_season": config.validation_season,
        "test_season": config.test_season,
        "split_rows": {key: len(frame) for key, frame in splits.items()},
        "split_dates": {key: {"first": str(frame.gameday.min().date()), "last": str(frame.gameday.max().date())} for key, frame in splits.items()},
        "ignored_input_rows": len(games) - sum(len(frame) for frame in splits.values()),
        "alpha_grid": sorted(config.alphas),
        "selected_alphas": selected_alphas,
        "selection_metric": f"{config.validation_season} pooled home/away score RMSE",
        "selection_tie_break": "Larger alpha",
        "final_fit_seasons": sorted([*config.train_seasons, config.validation_season]),
        "final_fit_rows": len(refit),
        "football_features": feature_groups()["football_features"],
        "market_features": feature_groups()["market_features"],
        "prediction_policy": "Nonnegative continuous home/away scores; total and margin are their sum and difference. Market benchmark preserves each available raw line.",
        "preprocessing": "Training-only median imputation, missing indicators, standard scaling, and one-hot categories with unknown categories ignored. All-missing numeric training columns retained with a zero fill.",
        "holdout_policy": "Alpha selected on validation only; refit training+validation once, then score test. No weekly coefficient refits. Prior-game features can advance within validation/test seasons as games finish.",
        "evaluation_policy": "Both available-case and common-game cohorts; MAE/RMSE/bias in points. Validation metrics are selection-biased; test is the holdout evaluation.",
        "limitations": [
            "This run estimates means, not calibrated distributions or betting probabilities.",
            "No ROI, Brier score, log loss, or CLV is claimed from point predictions.",
            "Recorded sportsbook lines lack decision timestamps; market comparisons are retrospective benchmarks.",
            "Historical input snapshots may contain later corrections and retrospective EPA/CPOE revisions.",
            "After inspecting 2025 results, further model changes need fresh future data or nested temporal evaluation; 2025 is no longer an unseen development holdout.",
        ],
    }
    return BaselineRun(predictions, evaluation_table(predictions), pd.concat(tuning_tables, ignore_index=True), final_models, metadata)
