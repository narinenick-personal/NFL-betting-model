"""Analytical settlement checks and chronological discrete-score experiments."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from nfl_model.discrete_scores import DiscreteScoreDistribution, ScoreGrid, _local_scoring_correction
from nfl_model.distribution_backtesting import MODEL_NAMES
from nfl_model.market_backtesting import market_specs, price_game, run_market_backtest
from nfl_model.market_reporting import save_market_run
from nfl_model.models import TrainingMeanModel
from nfl_model.probability_evaluation import event_metrics, summarize_probabilities
from nfl_model.settlement import (SettlementRules, american_profit, grade_market,
                                  implied_probability, no_vig_probability,
                                  overtime_format, price_comparison, unit_profit)
from nfl_model.team_games import TARGET_COLUMNS


@pytest.fixture
def cached_means(model_games):
    records, models = [], {}
    for season in range(2022, 2026):
        history = model_games.loc[model_games.season.lt(season)]
        forecast = model_games.loc[model_games.season.eq(season)]
        mean = TrainingMeanModel().fit(history)
        predicted = mean.predict(forecast)
        for name in MODEL_NAMES:
            for row in forecast.itertuples():
                records.append({"game_id": row.game_id, "season": season, "gameday": row.gameday,
                                "model": name, "mean_fit_end": history.gameday.max(),
                                "pred_home_score": predicted.loc[row.Index, "home_score"],
                                "pred_away_score": predicted.loc[row.Index, "away_score"],
                                "residual_home_score": row.home_score - predicted.loc[row.Index, "home_score"],
                                "residual_away_score": row.away_score - predicted.loc[row.Index, "away_score"]})
            if season == 2025:
                models[name] = SimpleNamespace(mean_model=mean, mean_fit_end=history.gameday.max())
    return pd.DataFrame(records), models


@pytest.fixture
def fitted_grid(model_games, cached_means):
    residuals, _ = cached_means
    return DiscreteScoreDistribution().fit(
        residuals.loc[residuals.season.lt(2025) & residuals.model.eq(MODEL_NAMES[0])],
        model_games.loc[model_games.season.lt(2025)], prediction_start="2025-09-01")


@pytest.mark.parametrize("market,side,line,expected", [
    ("moneyline", "home", None, [1, -1, 0]),
    ("moneyline", "away", None, [-1, 1, 0]),
    ("spread", "home", -3, [0, -1, -1]),
    ("spread", "away", -3, [0, 1, 1]),
    ("spread", "home", -2.5, [1, -1, -1]),
    ("spread", "home", 3, [1, 0, 1]),
    ("total", "over", 45, [0, 0, -1]),
    ("total", "under", 44.5, [-1, -1, 1]),
])
def test_settlement_known_results(market, side, line, expected):
    np.testing.assert_array_equal(grade_market([24, 21, 20], [21, 24, 20], market=market, side=side, line=line), expected)


def test_moneyline_tie_contract_and_rule_versions():
    for side in ("home", "away"):
        assert grade_market(20, 20, market="moneyline", side=side, rules=SettlementRules("loss")) == -1
    assert overtime_format(2024, "REG")["receiving_team_opening_td_ends_game"]
    assert not overtime_format(2025, "REG")["receiving_team_opening_td_ends_game"]
    assert overtime_format(2021, "DIV")["receiving_team_opening_td_ends_game"]
    assert not overtime_format(2022, "DIV")["receiving_team_opening_td_ends_game"]
    assert overtime_format(2025, "REG")["period_minutes"] == 10
    assert not overtime_format(2025, "SB")["final_tie_allowed"]
    with pytest.raises(ValueError, match="2021–2025"):
        overtime_format(2026, "REG")


@pytest.mark.parametrize("kwargs", [
    {"market": "spread", "side": "home", "line": -3.25},
    {"market": "total", "side": "home", "line": 40},
    {"market": "moneyline", "side": "home", "line": 0},
    {"market": "moneyline", "side": "home", "rules": SettlementRules(includes_overtime=False)},
])
def test_unsupported_contracts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        grade_market(20, 20, **kwargs)


def test_prices_and_push_aware_expected_profit():
    assert american_profit(150) == 1.5
    assert american_profit(-200) == .5
    assert implied_probability(-200) == pytest.approx(2 / 3)
    assert no_vig_probability(-110, -110) == .5
    np.testing.assert_allclose(unit_profit([1, 0, -1], -200), [.5, 0, -1])
    comparison = price_comparison(.6, .3, .1, -200, 180)
    assert comparison["expected_net_profit_per_unit"] == pytest.approx(0)
    assert comparison["edge_vs_price_conditional"] == pytest.approx(0)
    with pytest.raises(ValueError):
        american_profit(0)


def test_grid_exact_probabilities_and_sampling():
    grid = ScoreGrid(np.array([24, 21, 20, 28]), np.array([21, 24, 20, 20]), np.array([.3, .2, .1, .4]))
    assert grid.market_probabilities(market="moneyline", side="home") == pytest.approx({"p_win": .7, "p_loss": .2, "p_push": .1})
    assert grid.market_probabilities(market="spread", side="home", line=-3) == pytest.approx({"p_win": .4, "p_loss": .3, "p_push": .3})
    samples = grid.sample(draws=10_000, rng=np.random.default_rng(91))
    assert samples.home_margin.gt(0).mean() == pytest.approx(.7, abs=.015)
    np.testing.assert_array_equal(samples.game_total, samples.home_score + samples.away_score)
    np.testing.assert_array_equal(samples.home_margin, samples.home_score - samples.away_score)
    with pytest.raises(ValueError):
        ScoreGrid(np.array([2]), np.array([0]), np.array([1.000001]))


def test_nflverse_sign_conversion_and_tie_loss_benchmark(model_games):
    row = model_games.iloc[0].copy()
    row["market_spread_line"] = 3.0
    row = next(row.to_frame().T.itertuples(index=False))
    specs = market_specs(row)
    assert {s["line"] for s in specs if s["market"] == "spread"} == {-3.0}
    grid = ScoreGrid(np.array([24]), np.array([21]), np.array([1.0]))
    priced = price_game(row, grid, model="test", phase="test", rules=SettlementRules("loss"))
    for price in priced:
        if price["market"] == "spread":
            assert price["p_push"] == 1
        if price["market"] == "moneyline":
            assert np.isnan(price["no_vig_probability"])


def test_discrete_support_regular_ties_and_playoff_winner(fitted_grid):
    regular = fitted_grid.predict_grid([25, 21], season=2025, game_type="REG")
    playoff = fitted_grid.predict_grid([25, 21], season=2025, game_type="SB")
    assert regular.probabilities.sum() == pytest.approx(1)
    assert regular.market_probabilities(market="moneyline", side="home")["p_push"] > 0
    assert playoff.market_probabilities(market="moneyline", side="home")["p_push"] == 0
    assert not np.isin(1, regular.home)
    samples = playoff.sample(draws=10_000, rng=np.random.default_rng(93))
    assert (samples.home_score != samples.away_score).all()
    assert (samples.to_numpy() % 1 == 0).all()
    assert regular.market_probabilities(market="total", side="over", line=47.5)["p_push"] == 0
    with pytest.raises(ValueError, match="too small"):
        fitted_grid.predict_grid([99, 99], season=2025, game_type="REG")


def test_local_scoring_corrections_preserve_key_spikes():
    counts = np.zeros(51)
    counts[[3, 7, 14, 21, 28]] = 100
    correction = _local_scoring_correction(counts, 2)
    assert correction[21] > correction[20]
    assert correction.min() >= .25 and correction.max() <= 4


@pytest.mark.parametrize("problem", ["future", "playoff_tie", "rare_score", "wrong_residual", "wrong_date"])
def test_discrete_fitting_rejects_bad_history(model_games, cached_means, problem):
    errors = cached_means[0].loc[lambda d: d.season.lt(2025) & d.model.eq(MODEL_NAMES[0])].copy()
    history = model_games.loc[model_games.season.lt(2025)].copy()
    if problem == "future":
        history.loc[0, "gameday"] = pd.Timestamp("2025-09-01")
    elif problem == "playoff_tie":
        history.loc[0, "game_type"] = "DIV"
        history.loc[0, "home_score"] = history.loc[0, "away_score"]
    elif problem == "rare_score":
        history.loc[0, "home_score"] = 1
    elif problem == "wrong_residual":
        errors["residual_home_score"] += 1
    else:
        errors["gameday"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        DiscreteScoreDistribution().fit(errors, history, prediction_start="2025-09-01")


def test_binary_and_three_class_scoring_with_pushes():
    rows = pd.DataFrame({"outcome": [1, -1, 0], "p_win": [.6] * 3, "p_loss": [.3] * 3,
                         "p_push": [.1] * 3, "p_conditional": [2 / 3] * 3})
    scores = event_metrics(rows)
    assert scores["decisions"] == 2 and scores["pushes"] == 1
    assert scores["binary_brier"] == pytest.approx(5 / 18)
    assert scores["binary_log_loss"] == pytest.approx(-.5 * np.log(2 / 9))
    expected = np.mean(np.square(np.array([.6, .3, .1]) - np.eye(3)).sum(axis=1))
    assert scores["multiclass_brier"] == pytest.approx(expected)
    assert scores["multiclass_log_loss"] == pytest.approx(-np.log([.6, .3, .1]).mean())
    rows[["p_win", "p_loss", "p_push"]] = np.nan
    benchmark = event_metrics(rows)
    assert benchmark["unconditional_games"] == 0
    assert np.isnan(benchmark["multiclass_brier"])
    assert np.isnan(benchmark["mean_predicted_push_rate"])
    rows.loc[0, "p_win"] = .5
    with pytest.raises(ValueError, match="all unconditional"):
        event_metrics(rows)


def test_fold_cutoffs_sampling_and_canonical_metrics(model_games, cached_means):
    errors, models = cached_means
    run = run_market_backtest(model_games, errors, mean_models=models)
    assert len(run.predictions) == 24 * 4 * 6
    assert run.metrics.games.eq(8).all()
    assert run.metrics.decisions.le(8).all()
    assert run.calibration_bins.groupby(["phase", "season", "model", "market"]).games.sum().le(8).all()
    assert not run.predictions.duplicated(["game_id", "model", "market", "side"]).any()
    start = model_games.groupby("season").gameday.min()
    assert run.fit_log.calibration_end.lt(run.fit_log.season.map(start)).all()
    assert run.fit_log.score_history_end.lt(run.fit_log.season.map(start)).all()
    audit = model_games.loc[model_games.season.eq(2025)].drop(columns=list(TARGET_COLUMNS))
    for model in run.models.values():
        assert not set(audit.game_id).intersection(model.distribution.score_history_ids_)
        assert not set(audit.game_id).intersection(model.distribution.residual_model_.calibration_game_ids_)
    model = run.models[MODEL_NAMES[0]]
    first = dict(model.simulate(audit, draws=200, seed=5))
    second = dict(model.simulate(audit.sample(frac=1, random_state=23), draws=200, seed=5))
    for game_id in first:
        assert_frame_equal(first[game_id], second[game_id])
    with pytest.raises(ValueError, match="cutoffs"):
        list(model.grids(model_games.loc[model_games.season.eq(2024)]))


def test_audit_outcomes_do_not_change_probabilities(model_games, cached_means):
    errors, models = cached_means
    original = run_market_backtest(model_games, errors, mean_models=models)
    altered = model_games.copy()
    altered.loc[altered.season.eq(2025), "home_score"] += 4
    changed = run_market_backtest(altered, errors, mean_models=models)
    columns = ["game_id", "model", "market", "side", "p_win", "p_loss", "p_push", "p_conditional"]
    assert_frame_equal(original.predictions[columns], changed.predictions[columns])
    assert_frame_equal(original.fit_log, changed.fit_log)
    assert_frame_equal(original.metrics.loc[original.metrics.phase.eq("development")], changed.metrics.loc[changed.metrics.phase.eq("development")])


@pytest.mark.parametrize("problem", ["duplicate", "date", "season", "missing", "in_sample", "saved_model"])
def test_cached_projection_integrity(model_games, cached_means, problem):
    errors, models = cached_means
    errors = errors.copy()
    if problem == "duplicate":
        errors = pd.concat([errors, errors.iloc[:1]])
    elif problem == "date":
        errors.loc[0, "gameday"] += pd.Timedelta(days=1)
    elif problem == "season":
        errors.loc[0, "season"] = 2021
    elif problem == "missing":
        errors = errors.loc[~errors.game_id.eq("2024_01_A_B")]
    elif problem == "in_sample":
        errors.loc[0, "mean_fit_end"] = errors.loc[0, "gameday"]
    else:
        for item in models.values():
            item.mean_model = TrainingMeanModel().fit(model_games.iloc[:2])
    with pytest.raises(ValueError):
        run_market_backtest(model_games, errors, mean_models=models)


def test_loss_contract_omits_incomplete_no_vig_moneyline(model_games, cached_means):
    run = run_market_backtest(model_games, cached_means[0], rules=SettlementRules("loss"))
    assert not ((run.predictions.model == "no_vig_prices") & (run.predictions.market == "moneyline")).any()
    assert run.predictions.loc[run.predictions.market.eq("moneyline"), "p_push"].eq(0).all()


def test_invalid_audit_playoff_tie_is_rejected(model_games, cached_means):
    audit_index = model_games.index[model_games.season.eq(2025)][0]
    model_games.loc[audit_index, "game_type"] = "SB"
    model_games.loc[audit_index, "home_score"] = model_games.loc[audit_index, "away_score"]
    with pytest.raises(ValueError, match="postseason final"):
        run_market_backtest(model_games, cached_means[0])


def test_all_push_probability_has_no_conditional_metric():
    rows = pd.DataFrame({"outcome": [0], "p_win": [0.], "p_loss": [0.], "p_push": [1.], "p_conditional": [np.nan]})
    metrics = event_metrics(rows)
    assert metrics["multiclass_log_loss"] == 0
    assert np.isnan(metrics["binary_brier"])
    rows["p_conditional"] = .5
    with pytest.raises(ValueError, match="undefined"):
        event_metrics(rows)


def test_saved_market_artifacts(model_games, cached_means, tmp_path):
    errors, models = cached_means
    run = run_market_backtest(model_games, errors, mean_models=models)
    dataset = tmp_path / "games.parquet"
    model_games.to_parquet(dataset, index=False)
    report = save_market_run(run, tmp_path / "output", dataset_path=dataset, games=model_games,
                             source_paths={}, tests="fixture test", draws=200)
    assert "Retrospective audit" in report.read_text()
    metadata = json.loads((report.parent / "run.json").read_text())
    assert metadata["tests"] == "fixture test"
    assert len(metadata["artifact_sha256"]["models.joblib"]) == 64
    saved = joblib.load(report.parent / "models.joblib")
    assert set(saved) == set(MODEL_NAMES)
    simulations = pd.read_parquet(report.parent / "example_simulations.parquet")
    assert simulations.groupby("model").size().eq(200).all()
    assert (simulations[list(TARGET_COLUMNS)].to_numpy() % 1 == 0).all()
    checks = pd.read_parquet(report.parent / "simulation_checks.parquet")
    assert len(checks) == 8 * 3 * 3
    probabilities = pd.read_parquet(report.parent / "predictions.parquet")
    metrics, bins = summarize_probabilities(probabilities)
    assert_frame_equal(metrics, run.metrics)
    assert_frame_equal(bins, run.calibration_bins)
