import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nfl_model.distributions import JointResidualDistribution, game_rng
from nfl_model.distribution_backtesting import DistributionConfig, inner_temporal_split, run_distribution_backtest, select_distributions
from nfl_model.distribution_evaluation import ensemble_crps, energy_score, evaluate_draws, wilson_interval
from nfl_model.distribution_reporting import save_distribution_run
from nfl_model.team_games import TARGET_COLUMNS


@pytest.fixture
def paired_residuals():
    rng = np.random.default_rng(123)
    values = rng.multivariate_normal([1, -2], [[4, 5], [5, 9]], 300)
    return pd.DataFrame({
        "game_id": [f"game_{i}" for i in range(300)],
        "gameday": pd.date_range("2022-01-01", periods=300),
        "mean_fit_end": pd.Timestamp("2021-12-31"),
        "residual_home_score": values[:, 0], "residual_away_score": values[:, 1],
    })


@pytest.mark.parametrize("kind", ["paired_bootstrap", "gaussian"])
def test_joint_sampling_preserves_dependence_bias_and_algebra(paired_residuals, kind):
    distribution = JointResidualDistribution(kind).fit(paired_residuals, prediction_start=pd.Timestamp("2023-01-01"))
    samples = distribution.sample([100, 100], draws=30_000, rng=np.random.default_rng(1))
    assert samples.shape == (30_000, 4)
    assert np.corrcoef(samples[:, :2].T)[0, 1] > .65
    np.testing.assert_allclose(samples[:, :2].mean(axis=0), [100, 100] + distribution.bias_, atol=.08)
    np.testing.assert_allclose(samples[:, 2], samples[:, 0] + samples[:, 1])
    np.testing.assert_allclose(samples[:, 3], samples[:, 0] - samples[:, 1])
    assert np.linalg.eigvalsh(distribution.covariance_).min() > 0


def test_bootstrap_uses_whole_pairs_and_censors_negative_scores(paired_residuals):
    paired_residuals["residual_home_score"] = np.tile([-30., 4., 8.], 100)
    paired_residuals["residual_away_score"] = np.tile([-40., 12., 16.], 100)
    model = JointResidualDistribution().fit(paired_residuals, prediction_start="2023-01-01")
    samples = model.sample([20, 20], draws=1000, rng=game_rng(9, "game", "pairs"))
    assert set(map(tuple, samples[:, :2])) == {(0., 0.), (24., 32.), (28., 36.)}


@pytest.mark.parametrize("issue", ["in_sample", "future", "duplicate", "missing_date", "missing_residual", "too_few"])
def test_invalid_calibration_data_rejected(paired_residuals, issue):
    if issue == "in_sample":
        paired_residuals.loc[0, "mean_fit_end"] = paired_residuals.loc[0, "gameday"]
    elif issue == "future":
        paired_residuals.loc[0, "gameday"] = pd.Timestamp("2023-01-01")
    elif issue == "duplicate":
        paired_residuals.loc[0, "game_id"] = paired_residuals.loc[1, "game_id"]
    elif issue == "missing_date":
        paired_residuals.loc[0, "mean_fit_end"] = pd.NaT
    elif issue == "missing_residual":
        paired_residuals.loc[0, "residual_home_score"] = np.nan
    else:
        paired_residuals = paired_residuals.iloc[:2]
    with pytest.raises(ValueError):
        JointResidualDistribution().fit(paired_residuals, prediction_start="2023-01-01")


def test_crps_matches_explicit_pairwise_definition():
    samples = np.array([-3., -1., 0., 0., 2., 4.])
    actual = 1.0
    expected = np.abs(samples - actual).mean() - .5 * np.abs(samples[:, None] - samples[None, :]).mean()
    assert ensemble_crps(actual, samples) == pytest.approx(expected)
    assert ensemble_crps(0, np.array([-1., 0., 1.])) == pytest.approx(2 / 9)
    assert ensemble_crps(10, np.full(100, 10.)) == 0
    with pytest.raises(ValueError):
        ensemble_crps(10, np.array([np.nan]))


def test_energy_score_and_interval_metrics_known_values():
    scores = np.tile([10., 20.], (100, 1))
    assert energy_score(np.array([13., 24.]), scores) == 5
    samples = np.column_stack([scores, scores.sum(axis=1), scores[:, 0] - scores[:, 1]])
    rows, joint = evaluate_draws([10, 20, 30, -10], samples, pit_rng=np.random.default_rng(1))
    assert joint["energy_score"] == 0
    for row in rows:
        assert row["crps"] == 0
        assert row["covered_80"]
        assert row["width_80"] == 0
        assert 0 <= row["pit"] <= 1
    rows, _ = evaluate_draws([13, 24, 37, -11], samples, pit_rng=np.random.default_rng(1))
    assert rows[0]["interval_score_80"] == pytest.approx(30)
    assert not rows[0]["covered_80"]
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(.4038315303659956)
    assert high == pytest.approx(.5961684696340044)


def test_initial_and_later_inner_splits_never_use_outer_year(model_games):
    first = model_games.loc[model_games.season.eq(2021)]
    train, validation = inner_temporal_split(first)
    assert len(train) == 6 and len(validation) == 2
    assert train.gameday.max() < validation.gameday.min()
    train, validation = inner_temporal_split(model_games.loc[model_games.season.le(2023)])
    assert set(train.season) == {2021, 2022}
    assert set(validation.season) == {2023}


@pytest.fixture
def tiny_config():
    return DistributionConfig(alphas=(100.0,), draws=100, min_calibration_games=3)


def test_nested_forecasts_and_calibration_cutoffs(model_games, tiny_config):
    run = run_distribution_backtest(model_games, tiny_config)
    assert (run.residuals.mean_fit_end < run.residuals.gameday).all()
    assert (run.tuning.inner_train_end < run.tuning.inner_validation_start).all()
    assert (run.tuning.inner_validation_end <= run.tuning.mean_fit_end).all()
    dates = model_games.set_index("game_id").gameday
    assert (run.diagnostics.calibration_end < run.diagnostics.game_id.map(dates)).all()
    assert run.metadata["final_calibration_rows"] == {name: 24 for name in run.models}
    assert run.residuals.loc[run.residuals.season.eq(2025), "calibration_eligible"].eq(False).all()
    assert set(run.diagnostics.phase) == {"development", "retrospective_audit"}
    assert set(run.selection.games) == {16}
    assert run.pit_histograms.groupby(["phase", "season", "model", "distribution", "target"])["count"].sum().eq(8).all()
    assert not run.residuals.duplicated(["game_id", "model"]).any()
    for joint in run.models.values():
        audit_ids = set(model_games.loc[model_games.season.eq(2025), "game_id"])
        assert not audit_ids.intersection(joint.distribution.calibration_game_ids_)


def test_audit_outcomes_cannot_change_model_distribution_or_selection(model_games, tiny_config):
    original = run_distribution_backtest(model_games, tiny_config)
    changed = model_games.copy()
    audit = changed.season.eq(2025)
    changed.loc[audit, "home_score"] += 100
    changed.loc[audit, "away_score"] += 50
    changed["game_total"] = changed.home_score + changed.away_score
    changed["home_margin"] = changed.home_score - changed.away_score
    altered = run_distribution_backtest(changed, tiny_config)
    assert_frame_equal(original.selection, altered.selection)
    assert_frame_equal(original.tuning, altered.tuning)
    assert_frame_equal(original.metrics.loc[original.metrics.phase.eq("development")], altered.metrics.loc[altered.metrics.phase.eq("development")])
    np.testing.assert_allclose(original.diagnostics.sample_mean, altered.diagnostics.sample_mean, atol=1e-10)
    for name in original.models:
        first = original.models[name].distribution
        second = altered.models[name].distribution
        np.testing.assert_allclose(first.residuals_, second.residuals_, atol=1e-10)
        np.testing.assert_allclose(first.covariance_, second.covariance_, atol=1e-10)
        assert first.calibration_game_ids_ == second.calibration_game_ids_
    with pytest.raises(ValueError, match="development rows only"):
        select_distributions(original.joint_diagnostics)


def test_simulations_are_order_independent_and_target_free(model_games, tiny_config):
    run = run_distribution_backtest(model_games, tiny_config)
    audit = model_games.loc[model_games.season.eq(2025)].drop(columns=list(TARGET_COLUMNS))
    model = run.models["ridge_football"]
    first = dict(model.simulate(audit, draws=100, seed=123))
    second = dict(model.simulate(audit.sample(frac=1, random_state=9), draws=100, seed=123))
    for game_id in first:
        assert_frame_equal(first[game_id], second[game_id], atol=1e-10, rtol=1e-10)
    with pytest.raises(ValueError, match="cutoffs"):
        list(model.simulate(model_games.loc[model_games.season.eq(2024)]))


def test_save_and_reload_simulation_artifacts(model_games, tiny_config, tmp_path):
    run = run_distribution_backtest(model_games, tiny_config)
    dataset = tmp_path / "data.parquet"
    model_games.to_parquet(dataset, index=False)
    report = save_distribution_run(run, tmp_path / "outputs", dataset_path=dataset, games=model_games, tests="test run")
    assert "Retrospective audit" in report.read_text()
    metadata = json.loads((report.parent / "run.json").read_text())
    assert len(metadata["model_bundle_sha256"]) == 64
    assert metadata["tests"] == "test run"
    assert "reproduced" in metadata["reload_check"]
    examples = pd.read_parquet(report.parent / "example_simulations.parquet")
    assert examples.groupby("model").size().eq(tiny_config.draws).all()
    assert not examples.duplicated(["game_id", "model", "draw_id"]).any()
    assert set(joblib.load(report.parent / "models.joblib")) == set(run.models)
