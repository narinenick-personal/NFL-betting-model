from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from test_player_dataset import player_sources
from nfl_model.discrete_scores import DiscreteGameModel, DiscreteScoreDistribution
from nfl_model.models import TrainingMeanModel
from nfl_model.opportunity_models import PlayerOpportunityModel, TeamVolumeModel, VOLUME_TARGETS
from nfl_model.player_backtesting import run_player_simulation_backtest
from nfl_model.player_dataset import build_player_dataset


def test_player_backtest_audit_outcome_invariance(player_sources, model_games):
    players = build_player_dataset(**player_sources).players
    games = model_games.loc[model_games.week.le(4)].copy()
    games["game_id"] = games.apply(lambda r: f"{r.season}_{r.week}_A_B", axis=1)
    games["home_team"], games["away_team"] = "A", "B"
    for side in ("home", "away"):
        games[f"{side}_attempts"] = 30 + games.week
        games[f"{side}_carries"] = 20 + games.week
        games[f"{side}_sacks"] = 1 + games.week % 2
    teams = player_sources["team_games"].merge(player_sources["team_stats"], on=["game_id", "team"])
    score_lookup = games.set_index("game_id")
    teams["points_for"] = [score_lookup.loc[r.game_id, "home_score" if r.team == "A" else "away_score"] for r in teams.itertuples()]
    bundles, score_rows, volume_rows = {}, [], []
    for season in range(2022, 2026):
        history, forecast = games.loc[games.season.lt(season)], games.loc[games.season.eq(season)]
        mean = TrainingMeanModel().fit(history)
        volume = TeamVolumeModel("training_mean").fit(history)
        opp = PlayerOpportunityModel().fit(players.loc[players.season.lt(season)])
        bundles[season] = dict(volume=volume, players=opp, target_rate=.8, fit_end=history.gameday.max())
        scores, counts = mean.predict(forecast), volume.predict(forecast)
        for r in forecast.itertuples():
            score_rows.append(dict(game_id=r.game_id, season=season, gameday=r.gameday, mean_fit_end=history.gameday.max(),
                                   pred_home_score=scores.loc[r.Index, "home_score"], pred_away_score=scores.loc[r.Index, "away_score"],
                                   residual_home_score=r.home_score - scores.loc[r.Index, "home_score"], residual_away_score=r.away_score - scores.loc[r.Index, "away_score"]))
            for target in VOLUME_TARGETS:
                volume_rows.append(dict(game_id=r.game_id, season=season, gameday=r.gameday, fit_end=history.gameday.max(), target=target, actual=getattr(r, target), predicted=counts.loc[r.Index, target]))
    errors = pd.DataFrame(score_rows)
    discrete = DiscreteScoreDistribution().fit(errors.loc[errors.season.lt(2025)], games.loc[games.season.lt(2025)], prediction_start="2025-09-01")
    final_score = DiscreteGameModel(mean, discrete, games.loc[games.season.lt(2025)].gameday.max())
    args = dict(final_score_model=final_score, draws=100, seed=83)
    first = run_player_simulation_backtest(games, players, teams, errors, pd.DataFrame(volume_rows), bundles, **args)
    changed = players.copy()
    changed.loc[changed.season.eq(2025), "receiving_yards"] += 20
    second = run_player_simulation_backtest(games, changed, teams, errors, pd.DataFrame(volume_rows), bundles, **args)
    assert_frame_equal(first.fit_log, second.fit_log)
    assert_frame_equal(first.example_draws, second.example_draws)
    fields = ["season", "game_id", "player_id", "statistic", "mean", "p_over", "interval_mass_80"]
    assert_frame_equal(first.diagnostics[fields], second.diagnostics[fields])
    assert first.diagnostics.pit.between(0, 1).all()
    assert first.diagnostics.interval_mass_80.ge(.8 - 1e-9).all()
    assert not np.allclose(first.diagnostics.actual, second.diagnostics.actual)
