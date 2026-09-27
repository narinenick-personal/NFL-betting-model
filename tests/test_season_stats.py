import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from nfl_model.season_stats import prepare_season_stats, summarize_stats, save_snapshot, read_snapshot


@pytest.fixture
def stat_sources():
    schedule = pd.DataFrame([
        ["g1", 2026, 1, "REG", "2026-09-10", "20:15", "BAL", 20, "KC", 27],
        ["g2", 2026, 2, "REG", "2026-09-17", "20:15", "KC", 24, "BAL", 30],
        ["g3", 2026, 3, "REG", "2026-09-24", "20:15", "BAL", np.nan, "KC", np.nan],
        ["old", 2025, 22, "SB", "2026-02-08", "18:30", "NE", 13, "SEA", 29],
    ], columns=["game_id", "season", "week", "game_type", "gameday", "gametime", "away_team", "away_score", "home_team", "home_score"])
    teams, players = [], []
    for game in schedule.itertuples():
        for side, other in ((game.home_team, game.away_team), (game.away_team, game.home_team)):
            row = dict(game_id=game.game_id, season=game.season, week=game.week, season_type="REG" if game.game_type == "REG" else "POST",
                       team=side, opponent_team=other, completions=5, attempts=10 if game.week == 1 else 30,
                       passing_yards=100, carries=10, rushing_yards=50, receptions=5, receiving_yards=100)
            teams.append(row)
            players.append({**row, "player_id": "p1" if side == "KC" else "p2", "player_display_name": "Test Quarterback", "position": "QB"})
    return schedule, pd.DataFrame(teams), pd.DataFrame(players)


def test_current_season_schedule_pending_stats_and_calendar_year(stat_sources):
    schedule, teams, players = prepare_season_stats(*stat_sources)
    assert "g3" not in set(teams.game_id) | set(players.game_id)
    assert schedule.set_index("game_id").loc["g3", "status"] == "Scheduled"
    assert schedule.set_index("game_id").loc["old", "season"] == 2025
    assert len(teams.loc[teams.season.eq(2026)]) == 4
    totals = summarize_stats(teams.loc[teams.season.eq(2026)]).set_index("team")
    assert totals.loc["KC", "passing_yards"] == 200
    assert totals.loc["KC", "completion_pct"] == 25  # 10/40, not mean(5/10, 5/30).
    assert totals.loc["KC", "points_for"] == 51


@pytest.mark.parametrize("damage", ["duplicate", "opponent", "season", "week"])
def test_reject_inconsistent_source_rows(stat_sources, damage):
    schedule, teams, players = stat_sources
    if damage == "duplicate":
        teams = pd.concat([teams, teams.iloc[[0]]], ignore_index=True)
    else:
        teams.loc[0, {"opponent": "opponent_team", "season": "season", "week": "week"}[damage]] = "NO" if damage == "opponent" else 999
    with pytest.raises(ValueError):
        prepare_season_stats(schedule, teams, players)


def test_partial_coverage_trades_missing_ids_and_zero_denominator(stat_sources):
    schedule, teams, players = stat_sources
    teams = teams.loc[~(teams.game_id.eq("g2") & teams.team.eq("BAL"))]
    players.loc[players.game_id.eq("g2"), "player_id"] = "p1"
    players.loc[0, "player_id"] = None
    players.loc[:, "attempts"] = 0
    schedule, teams, players = prepare_season_stats(schedule, teams, players)
    assert schedule.set_index("game_id").loc["g2", "status"] == "Awaiting stats"
    summary = summarize_stats(players, players=True)
    assert summary.player_id.notna().all()
    assert set(summary.loc[summary.player_id.eq("p1") & summary.season.eq(2026), "team"]) == {"KC", "BAL"}
    assert summary.completion_pct.isna().all()


def test_snapshot_atomic_roundtrip_and_failed_write_preserves_previous(stat_sources, tmp_path, monkeypatch):
    frames = prepare_season_stats(*stat_sources)
    path = tmp_path / "stats.zip"
    save_snapshot(path, *frames, {"seasons": [2025, 2026]})
    for expected, actual in zip(frames, read_snapshot(path)[:3]):
        assert_frame_equal(expected, actual)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    def fail(*args, **kwargs):
        raise ValueError("write failure")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(ValueError, match="write failure"):
        save_snapshot(path, *frames, {})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert len(list(tmp_path.glob("*.zip"))) == 1


def test_stats_ui_defaults_to_2026_filters_empty_weeks_and_postseason(stat_sources, tmp_path):
    save_snapshot(tmp_path / "data/season_stats/stats.zip", *prepare_season_stats(*stat_sources), {"built_at_utc": "2026-09-27T12:00:00Z"})
    app = AppTest.from_string(f"from pathlib import Path\nfrom nfl_model.dashboard import render_dashboard\nrender_dashboard(Path({str(tmp_path)!r}))", default_timeout=20).run()
    app.radio(key="view").set_value("Season stats").run()
    assert not app.exception and app.selectbox(key="stats_season").value == 2026
    assert app.metric[0].value == "2"
    app.selectbox(key="stats_team").set_value("KC").run()
    assert not app.exception and app.metric[1].value == "2"
    app.text_input(key="stats_search").set_value("Nobody").run()
    assert not app.exception
    app.multiselect(key="stats_weeks_2026_Regular season").set_value([3]).run()
    assert not app.exception and app.metric[0].value == "0"
    app.radio(key="stats_phase").set_value("Postseason").run()
    assert not app.exception
    app.selectbox(key="stats_season").set_value(2025).run()
    assert not app.exception and app.metric[0].value == "1"


def test_cli_refresh_preserves_other_seasons_and_previous_snapshot_on_failure(stat_sources, tmp_path, monkeypatch):
    from nfl_model import data_loader
    monkeypatch.setenv("NFLREADPY_CACHE", "memory")
    script = Path(__file__).resolve().parents[1] / "scripts/refresh_season_stats.py"
    spec = importlib.util.spec_from_file_location("refresh_stats_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = tmp_path / "raw"
    cache.mkdir()
    monkeypatch.setattr(data_loader, "DEFAULT_CACHE_DIR", cache)
    path = tmp_path / "stats.zip"
    old = [frame.loc[frame.season.eq(2025)] for frame in prepare_season_stats(*stat_sources)]
    save_snapshot(path, *old, {"sources": {}})
    calls = []
    for name, frame in zip(("schedules", "team_stats", "player_stats"), stat_sources):
        def load(seasons, *, refresh, name=name, frame=frame):
            calls.append((name, seasons, refresh))
            selected = frame.loc[frame.season.isin(seasons)]
            selected.to_parquet(cache / f"{name}_{seasons[0]}.parquet", index=False)
            return selected
        monkeypatch.setattr(data_loader, "load_" + name, load)
    monkeypatch.setattr(sys, "argv", [str(script), "--seasons", "2026", "--refresh", "--output", str(path)])
    module.main()
    refreshed = read_snapshot(path)
    assert refreshed[3]["seasons"] == [2025, 2026]
    assert all(refresh for _, _, refresh in calls)
    for prior, after in zip(old, refreshed[:3]):
        assert_frame_equal(prior.reset_index(drop=True), after.loc[after.season.eq(2025)].reset_index(drop=True))
    before = path.read_bytes()
    def fail(*args, **kwargs):
        raise ConnectionError("unavailable")
    monkeypatch.setattr(data_loader, "load_player_stats", fail)
    with pytest.raises(ConnectionError):
        module.main()
    assert path.read_bytes() == before
