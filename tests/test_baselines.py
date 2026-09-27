import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nfl_model.backtesting import BacktestConfig, chronological_splits, run_baseline_backtest, select_ridge_alpha
from nfl_model.evaluation import evaluation_table, regression_metrics
from nfl_model.model_reporting import save_baseline_run, verify_saved_predictions
from nfl_model.modeling_dataset import feature_groups
from nfl_model.models import MarketLineModel, RecentTeamAverageModel, RidgeScoreModel, TrainingMeanModel, score_predictions
from nfl_model.team_games import TARGET_COLUMNS



def test_chronological_splits_use_nfl_season_and_dates(model_games):
    model_games.loc[7, "gameday"] = pd.Timestamp("2022-01-15")
    split = chronological_splits(model_games.sample(frac=1, random_state=12), BacktestConfig())
    assert {name: len(frame) for name, frame in split.items()} == {"train": 24, "validation": 8, "test": 8}
    assert set(split["train"].season) == {2021, 2022, 2023}
    assert set(split["validation"].season) == {2024}
    assert set(split["test"].season) == {2025}
    assert model_games.loc[7, "game_id"] in set(split["train"].game_id)
    assert split["train"].gameday.max() < split["validation"].gameday.min()
    assert split["validation"].gameday.max() < split["test"].gameday.min()


@pytest.mark.parametrize("config", [
    BacktestConfig(train_seasons=()), BacktestConfig(train_seasons=(2023, 2023)),
    BacktestConfig(train_seasons=(2024,)), BacktestConfig(test_season=2024),
    BacktestConfig(alphas=()), BacktestConfig(alphas=(0.0,)),
    BacktestConfig(alphas=(np.nan,)), BacktestConfig(alphas=(1.0, 1.0)),
])
def test_invalid_protocol_rejected(model_games, config):
    with pytest.raises(ValueError):
        chronological_splits(model_games, config)


@pytest.mark.parametrize("issue", ["duplicate", "missing_season", "overlapping_date", "inconsistent_target", "missing_target"])
def test_bad_dataset_rejected(model_games, issue):
    if issue == "duplicate":
        model_games = pd.concat([model_games, model_games.iloc[:1]])
    elif issue == "missing_season":
        model_games = model_games.loc[model_games.season.ne(2024)]
    elif issue == "overlapping_date":
        model_games.loc[0, "gameday"] = pd.Timestamp("2026-01-01")
    elif issue == "inconsistent_target":
        model_games.loc[0, "game_total"] = 1000
    else:
        model_games.loc[0, "away_score"] = np.nan
    with pytest.raises(ValueError):
        chronological_splits(model_games, BacktestConfig())


def test_preprocessing_fits_training_only_handles_missing_and_unknown_categories(model_games):
    train = model_games.loc[model_games.season.le(2023)].copy()
    later = model_games.loc[model_games.season.eq(2024)].copy()
    feature = "home_pregame_points_for_last_3"
    train[feature] = np.arange(len(train), dtype=float)
    train.loc[train.index[0], feature] = np.nan
    later[feature] = 1_000_000.0
    all_missing = "away_pregame_cpoe_last_3"
    train[all_missing] = np.nan
    later[all_missing] = 0.3
    later["surface"] = "never_seen_before"
    later.loc[later.index[0], "game_type"] = pd.NA
    model = RidgeScoreModel(alpha=100).fit(train)
    transformer = model.pipeline_.named_steps["preprocess"]
    numeric_cols = transformer.transformers_[0][2]
    imputer = transformer.named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[numeric_cols.index(feature)] == train[feature].median()
    assert imputer.statistics_[numeric_cols.index(all_missing)] == 0
    medians_before = imputer.statistics_.copy()
    result = model.predict(later)
    assert np.isfinite(result).all().all()
    np.testing.assert_array_equal(imputer.statistics_, medians_before)
    encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
    assert "never_seen_before" not in encoder.categories_[1]
    assert (result.home_score >= 0).all()
    np.testing.assert_allclose(result.game_total, result.home_score + result.away_score)
    np.testing.assert_allclose(result.home_margin, result.home_score - result.away_score)


def test_football_predictions_ignore_markets_observations_and_targets(model_games):
    train = model_games.loc[model_games.season.le(2023)]
    evaluation = model_games.loc[model_games.season.eq(2024)].copy()
    model = RidgeScoreModel().fit(train)
    expected = model.predict(evaluation)
    excluded = [*TARGET_COLUMNS, *feature_groups()["market_features"], "temp", "wind", "roof"]
    actual = model.predict(evaluation.drop(columns=excluded))
    assert_frame_equal(actual, expected)
    assert not set(excluded).intersection(model.feature_names_)
    market = RidgeScoreModel(include_market=True).fit(train)
    assert set(feature_groups()["market_features"]).issubset(market.feature_names_)
    with pytest.raises(ValueError, match="missing expected"):
        market.predict(evaluation.drop(columns="market_total_line"))


def test_training_and_recent_benchmarks_use_past_fallbacks(model_games):
    train = model_games.loc[model_games.season.le(2023)]
    test = model_games.loc[model_games.season.eq(2025)].copy()
    mean = TrainingMeanModel().fit(train)
    np.testing.assert_allclose(mean.predict(test).home_score, train.home_score.mean())
    recent = RecentTeamAverageModel().fit(train)
    assert recent.predict(test).iloc[0].home_score == 23
    assert recent.predict(test).iloc[0].away_score == 18
    columns = ["home_pregame_points_for_last_5", "away_pregame_points_against_last_5", "away_pregame_points_for_last_5", "home_pregame_points_against_last_5"]
    test[columns] = np.nan
    assert_frame_equal(recent.predict(test), mean.predict(test))


def test_market_spread_sign_and_missing_line_coverage():
    games = pd.DataFrame({"market_total_line": [44.0, 40.0, np.nan, 42.0], "market_spread_line": [4.0, -6.0, 3.0, np.nan]})
    result = MarketLineModel().predict(games)
    assert result.loc[0].to_dict() == {"home_score": 24, "away_score": 20, "game_total": 44, "home_margin": 4}
    assert result.loc[1, "home_score"] == 17
    assert result.loc[1, "away_score"] == 23
    assert result.loc[2, "home_margin"] == 3
    assert pd.isna(result.loc[2, "game_total"])
    assert result.loc[3, "game_total"] == 42
    assert pd.isna(result.loc[3, "home_score"])


def test_metrics_have_correct_units_and_bias():
    metrics = regression_metrics(np.array([10, 20]), np.array([12, 16]))
    assert metrics["mae"] == 3
    assert metrics["rmse"] == pytest.approx(np.sqrt(10))
    assert metrics["bias"] == -1
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, 2], [1, np.nan])


def test_tuning_selects_validation_rmse_minimum(model_games):
    split = chronological_splits(model_games, BacktestConfig())
    model, table = select_ridge_alpha(split["train"], split["validation"], include_market=False, alphas=(1.0, 100.0, 10000.0))
    assert table.selected.sum() == 1
    assert table.loc[table.selected, "score_rmse"].iloc[0] == table.score_rmse.min()
    assert model.alpha == table.loc[table.selected, "alpha"].iloc[0]
    assert model.training_seasons_ == [2021, 2022, 2023]


def test_test_outcomes_cannot_affect_selection_fit_or_predictions(model_games):
    config = BacktestConfig(alphas=(10.0, 1000.0))
    original = run_baseline_backtest(model_games, config)
    altered = model_games.copy()
    test = altered.season.eq(2025)
    altered.loc[test, "home_score"] += 100
    altered.loc[test, "away_score"] += 50
    altered.loc[test, "game_total"] = altered.home_score + altered.away_score
    altered.loc[test, "home_margin"] = altered.home_score - altered.away_score
    changed = run_baseline_backtest(altered, config)
    assert original.metadata["selected_alphas"] == changed.metadata["selected_alphas"]
    assert_frame_equal(original.tuning, changed.tuning)
    columns = ["game_id", "split", "model", *[f"pred_{t}" for t in TARGET_COLUMNS]]
    assert_frame_equal(original.predictions[columns], changed.predictions[columns])
    assert_frame_equal(original.metrics.loc[original.metrics.split.eq("validation")], changed.metrics.loc[changed.metrics.split.eq("validation")])
    for name in ("ridge_football", "ridge_market"):
        assert original.models[name].training_seasons_ == [2021, 2022, 2023, 2024]
        assert original.models[name].training_rows_ == 32
        np.testing.assert_allclose(original.models[name].pipeline_.named_steps["regressor"].coef_, changed.models[name].pipeline_.named_steps["regressor"].coef_, rtol=1e-12, atol=1e-12)


def test_common_cohort_is_identical_across_models(model_games):
    model_games.loc[model_games.season.eq(2025).idxmax(), "market_total_line"] = np.nan
    run = run_baseline_backtest(model_games, BacktestConfig(alphas=(100.0,)))
    test_metrics = run.metrics.loc[run.metrics.split.eq("test")]
    total_common = test_metrics.loc[test_metrics.target.eq("game_total") & test_metrics.cohort.eq("common")]
    assert total_common.scored_games.eq(7).all()
    margin_common = test_metrics.loc[test_metrics.target.eq("home_margin") & test_metrics.cohort.eq("common")]
    assert margin_common.scored_games.eq(8).all()
    total_available = test_metrics.loc[test_metrics.target.eq("game_total") & test_metrics.cohort.eq("available")].set_index("model")
    assert total_available.loc["market_lines", "scored_games"] == 7
    assert total_available.loc["ridge_football", "scored_games"] == 8
    assert not run.predictions.duplicated(["split", "model", "game_id"]).any()


def test_no_common_market_coverage_reported_as_unscored(model_games):
    model_games["market_total_line"] = np.nan
    run = run_baseline_backtest(model_games, BacktestConfig(alphas=(100.0,)))
    common_totals = run.metrics.loc[run.metrics.target.eq("game_total") & run.metrics.cohort.eq("common")]
    assert common_totals.scored_games.eq(0).all()
    assert common_totals.rmse.isna().all()


def test_serialized_bundle_reproduces_holdout_and_prediction_alignment(model_games, tmp_path):
    games = model_games.sample(frac=1, random_state=42).reset_index(drop=True)
    run = run_baseline_backtest(games, BacktestConfig(alphas=(100.0,)))
    data_path = tmp_path / "games.parquet"
    games.to_parquet(data_path, index=False)
    report_path = save_baseline_run(run, tmp_path / "output", dataset_path=data_path, tests="test run")
    verify_saved_predictions(run, games, report_path.parent)
    assert_frame_equal(pd.read_parquet(report_path.parent / "predictions.parquet"), run.predictions)
    metadata = json.loads((report_path.parent / "run.json").read_text())
    assert metadata["split_rows"] == {"train": 24, "validation": 8, "test": 8}
    assert len(metadata["dataset"]["sha256"]) == 64
    assert metadata["tests"] == "test run"
    assert "Test MAE" in report_path.read_text()
    reloaded = joblib.load(report_path.parent / "models.joblib")
    assert set(reloaded) == set(run.models)


def test_training_script_runs_offline_with_custom_paths(model_games, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_baselines.py"
    spec = importlib.util.spec_from_file_location("training_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_path = tmp_path / "games.parquet"
    model_games.to_parquet(data_path, index=False)
    output = tmp_path / "results"
    module.main(["--dataset", str(data_path), "--output-dir", str(output), "--alphas", "100"])
    assert (output / "report.md").exists()
    saved = pd.read_parquet(output / "predictions.parquet")
    assert len(saved) == 2 * 8 * 5
    assert set(saved.split) == {"validation", "test"}


def test_evaluation_rejects_duplicate_predictions(model_games):
    run = run_baseline_backtest(model_games, BacktestConfig(alphas=(100.0,)))
    with pytest.raises(ValueError, match="duplicate keys"):
        evaluation_table(pd.concat([run.predictions, run.predictions.iloc[:1]]))


def test_negative_model_means_clip_before_deriving_total_margin():
    predictions = score_predictions(np.array([[-3, 21], [28, -2]]), pd.Index([0, 1]))
    assert predictions.loc[0].to_dict() == {"home_score": 0, "away_score": 21, "game_total": 21, "home_margin": -21}
    assert predictions.loc[1].to_dict() == {"home_score": 28, "away_score": 0, "game_total": 28, "home_margin": 28}
