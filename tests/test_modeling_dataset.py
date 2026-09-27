import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nfl_model.features import add_pregame_features, add_team_rolling_features
from nfl_model.modeling_dataset import build_modeling_dataset, select_model_features
from nfl_model.team_games import build_team_game_table
from nfl_model.team_metrics import RATE_METRICS, TEAM_METRICS


def test_unique_rows_scores_opponents_and_targets(sample_data):
    stats, schedules = sample_data
    result = build_modeling_dataset(stats, schedules)
    assert len(result.games) == len(schedules)
    assert len(result.team_games) == 2 * len(schedules)
    assert not result.games.duplicated("game_id").any()
    assert not result.team_games.duplicated(["game_id", "team"]).any()
    for row in schedules.itertuples(index=False):
        teams = result.team_games.set_index(["game_id", "team"])
        for side, other in (("home", "away"), ("away", "home")):
            record = teams.loc[(row.game_id, getattr(row, f"{side}_team"))]
            assert record.points_for == getattr(row, f"{side}_score")
            assert record.points_against == getattr(row, f"{other}_score")
            assert record.opponent_team == getattr(row, f"{other}_team")
            assert record.is_home == int(side == "home")
        game = result.games.set_index("game_id").loc[row.game_id]
        assert game.home_score == row.home_score
        assert game.away_score == row.away_score
        assert game.game_total == row.home_score + row.away_score
        assert game.home_margin == row.home_score - row.away_score


def test_correct_home_away_history_and_neutral_context(sample_data):
    games = build_modeling_dataset(*sample_data).games.set_index("game_id")
    assert games.loc["g2", "away_pregame_points_for_last_3"] == 10
    assert pd.isna(games.loc["g2", "home_pregame_points_for_last_3"])
    assert games.loc["g3", "home_pregame_points_for_last_3"] == 25
    assert games.loc["g3", "away_pregame_points_for_last_3"] == 30
    assert games.loc["g3", "home_pregame_points_against_last_3"] == 25
    assert games.loc["g7", "neutral_site"] == 1
    assert games.loc["g7", "home_is_home"] == 1
    assert games.loc["g7", "away_is_home"] == 0
    assert games.loc["g3", "home_rest"] == 7
    assert games.loc["g3", "away_rest"] == 8


def test_rolling_windows_byes_postseason_and_season_reset(sample_data):
    games = build_modeling_dataset(*sample_data).games.set_index("game_id")
    assert games.loc["g6", "away_pregame_points_for_last_3"] == pytest.approx((50 + 80 + 90) / 3)
    assert games.loc["g6", "away_pregame_points_for_last_5"] == 54
    assert games.loc["g6", "away_pregame_points_for_season_avg"] == 54
    assert games.loc["g6", "away_pregame_games_played"] == 5
    assert games.loc["g6", "away_pregame_games_last_3"] == 3
    assert games.loc["g7", "home_pregame_games_played"] == 6
    assert games.loc["g7", "home_pregame_points_for_last_3"] == pytest.approx((80 + 90 + 120) / 3)
    assert games.loc["g8", "home_pregame_games_played"] == 0
    for metric in TEAM_METRICS:
        for suffix in ("last_3", "last_5", "season_avg"):
            assert pd.isna(games.loc["g8", f"home_pregame_{metric}_{suffix}"])


def test_current_and_future_outcomes_cannot_change_earlier_features(sample_data):
    stats, schedules = sample_data
    original = build_modeling_dataset(stats, schedules).games
    cutoff = "2024-09-22"
    ids = schedules.loc[schedules.gameday.ge(cutoff), "game_id"]
    changed_stats = stats.copy()
    measurements = [c for c in stats.select_dtypes(include="number") if c not in {"week", "season"}]
    changed_stats.loc[changed_stats.game_id.isin(ids), measurements] = 999
    changed_schedule = schedules.copy()
    changed_schedule.loc[changed_schedule.game_id.isin(ids), ["home_score", "away_score"]] = [123, 456]
    altered = build_modeling_dataset(changed_stats, changed_schedule).games
    before = original.gameday.le(pd.Timestamp(cutoff))
    assert_frame_equal(select_model_features(original.loc[before]), select_model_features(altered.loc[before]))
    # A changed past outcome should affect the following game's history.
    assert original.set_index("game_id").loc["g4", "away_pregame_points_for_last_3"] != altered.set_index("game_id").loc["g4", "away_pregame_points_for_last_3"]


def test_prefix_build_and_shuffled_input_equal_full_history(sample_data):
    stats, schedules = sample_data
    original = build_modeling_dataset(stats, schedules).games
    shuffled = build_modeling_dataset(stats.sample(frac=1, random_state=1), schedules.sample(frac=1, random_state=2)).games
    assert_frame_equal(original, shuffled)
    prefix_schedule = schedules.loc[schedules.gameday.le("2024-09-22")]
    prefix = build_modeling_dataset(stats.loc[stats.game_id.isin(prefix_schedule.game_id)], prefix_schedule).games
    assert_frame_equal(original.iloc[:3].reset_index(drop=True), prefix)


def test_game_dates_take_precedence_over_week_labels(sample_data):
    stats, schedules = sample_data
    # A postponed lower-numbered week must not be treated as already played.
    schedules.loc[schedules.game_id.eq("g2"), "week"] = 8
    stats.loc[stats.game_id.eq("g2"), "week"] = 8
    games = build_modeling_dataset(stats, schedules).games.set_index("game_id")
    assert games.loc["g3", "home_pregame_points_for_last_3"] == 25


def test_raw_and_derived_metric_definitions(sample_data):
    raw = build_team_game_table(*sample_data).iloc[0]
    expected = {
        "pass_attempts": 30, "rush_attempts": 20, "total_offensive_plays": 52,
        "pass_dropbacks": 32, "passing_yards_per_attempt": 8,
        "rushing_yards_per_carry": 5, "passing_epa_per_attempt": .3,
        "passing_epa_per_dropback": 9 / 32, "rushing_epa_per_carry": -.1,
        "cpoe": 4, "sack_rate": 2 / 32, "interception_rate": 1 / 30,
        "offensive_fumbles_lost": 2, "team_fumbles_lost": 4,
        "fumble_lost_rate": 2 / 52, "turnovers": 3, "turnover_rate": 3 / 52,
        "offensive_first_down_rate": 17 / 52, "explosive_passing_rate": .1,
        "explosive_rushing_rate": .2, "penalties_per_game": 6,
        "penalty_yards_per_game": 45, "offensive_scoring_efficiency": 3 / 52,
        "defensive_sacks": 3, "defensive_interceptions": 2,
    }
    for name, value in expected.items():
        assert raw[name] == pytest.approx(value), name


def test_zero_denominators_and_unavailable_sources_remain_missing(sample_data):
    stats, schedules = sample_data
    stats = stats.drop(columns=["passing_20", "rushing_10", "passing_cpoe", "receiving_fumbles_lost"])
    stats.loc[:, ["attempts", "carries", "sacks_suffered"]] = 0
    result = build_modeling_dataset(stats, schedules)
    assert result.team_games[list(RATE_METRICS)].isna().all().all()
    assert result.team_games[["cpoe", "offensive_fumbles_lost", "turnovers"]].isna().all().all()
    assert not np.isinf(result.games.select_dtypes(include="number").to_numpy(dtype=float, na_value=np.nan)).any()


def test_missing_prior_measurement_is_not_backfilled(sample_data):
    stats, schedules = sample_data
    stats["passing_cpoe"] = stats["passing_cpoe"].astype(float)
    stats.loc[stats.game_id.eq("g1"), "passing_cpoe"] = np.nan
    games = build_modeling_dataset(stats, schedules).games.set_index("game_id")
    assert pd.isna(games.loc["g2", "away_pregame_cpoe_last_3"])
    assert games.loc["g3", "home_pregame_cpoe_last_3"] == 4


@pytest.mark.parametrize("which", ["stats", "schedules"])
def test_duplicate_input_keys_fail(sample_data, which):
    stats, schedules = sample_data
    if which == "stats":
        stats = pd.concat([stats, stats.iloc[:1]])
    else:
        schedules = pd.concat([schedules, schedules.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate keys"):
        build_modeling_dataset(stats, schedules)


@pytest.mark.parametrize("column,value", [("opponent_team", "WRONG"), ("season", 1999), ("week", 99), ("season_type", "POST")])
def test_mismatched_schedule_metadata_fails(sample_data, column, value):
    stats, schedules = sample_data
    stats.loc[0, column] = value
    with pytest.raises(ValueError, match=f"{column} disagrees"):
        build_modeling_dataset(stats, schedules)


def test_missing_completed_game_stats_fail(sample_data):
    stats, schedules = sample_data
    with pytest.raises(ValueError, match="Unmatched completed"):
        build_modeling_dataset(stats.iloc[1:], schedules)


def test_unknown_game_and_incorrect_team_fail(sample_data):
    stats, schedules = sample_data
    unknown = stats.copy()
    unknown.loc[0, "game_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown schedule"):
        build_modeling_dataset(unknown, schedules)
    stats.loc[0, "team"] = "WRONG"
    with pytest.raises(ValueError, match="Unmatched completed"):
        build_modeling_dataset(stats, schedules)


def test_unplayed_games_excluded_but_partial_scores_rejected(sample_data):
    stats, schedules = sample_data
    schedules[["home_score", "away_score"]] = schedules[["home_score", "away_score"]].astype(float)
    schedules.loc[schedules.game_id.eq("g8"), ["home_score", "away_score"]] = np.nan
    result = build_modeling_dataset(stats, schedules)
    assert "g8" not in set(result.games.game_id)
    assert "g8" not in set(result.team_games.game_id)
    schedules.loc[schedules.game_id.eq("g8"), "home_score"] = 0
    with pytest.raises(ValueError, match="partially missing"):
        build_modeling_dataset(stats, schedules)


def test_explicit_feature_contract_excludes_results_market_and_observed_weather(sample_data):
    stats, schedules = sample_data
    schedules = schedules.assign(total=999, result=-999, overtime=1, home_qb_name="Future QB")
    result = build_modeling_dataset(stats, schedules)
    football = select_model_features(result.games)
    forbidden = {"home_score", "away_score", "game_total", "home_margin", "total", "result", "overtime", "home_qb_name", "temp", "wind", "roof"}
    assert not forbidden.intersection(football.columns)
    assert not any(c.startswith("market_") for c in football)
    assert not any(c.endswith("_x") or c.endswith("_y") for c in result.games)
    assert "home_points_for" not in result.games
    market = select_model_features(result.games, include_market=True)
    assert set(market) - set(football) == set(result.column_groups["market_features"])
    assert result.games.loc[0, "market_spread_line"] == 3
    assert result.games.loc[0, "game_total"] == 30
    assert result.games.loc[0, "home_margin"] == -10
    columns = [col for group in result.column_groups.values() for col in group]
    assert len(columns) == len(set(columns)) == len(result.games.columns)


def test_optional_context_not_fabricated(sample_data):
    stats, schedules = sample_data
    result = build_modeling_dataset(stats, schedules.drop(columns=["location", "temp", "roof", "surface"]))
    assert result.games[["neutral_site", "temp", "roof", "surface"]].isna().all().all()


def test_legacy_rolling_resets_at_season_boundary():
    df = pd.DataFrame({"team": ["A", "A"], "season": [2024, 2025], "week": [18, 1], "points": [99, 10]})
    assert add_team_rolling_features(df)["points_rolling_4"].isna().all()


@pytest.mark.parametrize("windows", [(0,), (-1,), (3, 3), (), (True,), (1.5,)])
def test_invalid_rolling_windows_fail(sample_data, windows):
    with pytest.raises(ValueError, match="positive integers"):
        build_modeling_dataset(*sample_data, windows=windows)


def test_same_team_same_day_rejected(sample_data):
    stats, schedules = sample_data
    schedules.loc[schedules.game_id.eq("g2"), "gameday"] = "2024-09-01"
    with pytest.raises(ValueError, match="duplicate keys"):
        build_modeling_dataset(stats, schedules)


def test_duplicate_history_records_rejected(sample_data):
    team_games = build_team_game_table(*sample_data)
    with pytest.raises(ValueError, match="duplicate keys"):
        add_pregame_features(pd.concat([team_games, team_games.iloc[:1]]), TEAM_METRICS)


def test_build_script_writes_parquet_manifest_and_missing_report(sample_data, monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_modeling_dataset.py"
    spec = importlib.util.spec_from_file_location("build_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats, schedules = sample_data
    monkeypatch.setattr(module, "load_schedules", lambda *a, **kw: schedules)
    monkeypatch.setattr(module, "load_team_stats", lambda *a, **kw: stats)
    monkeypatch.setattr(module, "DEFAULT_CACHE_DIR", tmp_path)
    for name, frame in (("schedules", schedules), ("team_stats", stats)):
        frame.to_parquet(tmp_path / f"{name}_2024_2025.parquet", index=False)
    output = tmp_path / "processed"
    module.main(["--seasons", "2024", "2025", "--output-dir", str(output)])
    saved = pd.read_parquet(output / "game_modeling_dataset.parquet")
    expected = build_modeling_dataset(stats, schedules)
    assert_frame_equal(saved, expected.games)
    assert_frame_equal(pd.read_parquet(output / "team_game_dataset.parquet"), expected.team_games)
    manifest = json.loads((output / "game_modeling_manifest.json").read_text())
    assert manifest["column_groups"] == expected.column_groups
    assert len(manifest["source_files"]["team_stats"]["sha256"]) == 64
    report = json.loads((output / "game_modeling_report.json").read_text())
    assert report["games"] == 8
    assert report["missing_pct_by_column"]["home_score"] == 0
    assert report["missing_pct_by_column"]["market_under_odds"] == 100
    assert report["duplicate_games"] == 0
    assert (output / "game_modeling_report.txt").exists()
