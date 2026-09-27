import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from nfl_model.player_dataset import build_player_dataset, PLAYER_FEATURES, PLAYER_TARGETS, OTHER
from nfl_model.opportunity_models import PlayerOpportunityModel, TeamVolumeModel, VOLUME_TARGETS


@pytest.fixture
def player_sources():
    rosters, stats, snaps, teams, totals, identities = [], [], [], [], [], []
    for season in range(2021, 2026):
        for week in range(1, 5):
            game_id = f"{season}_{week}_A_B"
            for team, opponent in (("A", "B"), ("B", "A")):
                teams.append(dict(game_id=game_id, team=team, opponent_team=opponent, season=season, week=week, game_type="REG", season_type="REG", gameday=pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)), is_home=int(team == "A")))
                team_total = dict(game_id=game_id, team=team, **{c: 0 for c in PLAYER_TARGETS})
                for i, position in enumerate(("QB", "RB", "WR", "TE")):
                    player_id, pfr = f"{team}{i}", f"pfr{team}{i}"
                    identities.append(dict(gsis_id=player_id, pfr_id=pfr))
                    rosters.append(dict(season=season, week=week, game_type="REG", team=team, gsis_id=player_id, full_name=player_id, position=position, pfr_id=pfr, status="ACT"))
                    row = dict(game_id=game_id, team=team, player_id=player_id, season=season, week=week, season_type="REG", opponent_team=opponent, position=position, player_display_name=player_id, **{c: 0 for c in PLAYER_TARGETS})
                    if i == 0:
                        row.update(attempts=30 + week, completions=18, passing_yards=200, passing_tds=1)
                    elif i == 1:
                        row.update(carries=20 + week, rushing_yards=80, rushing_tds=1, targets=5, receptions=3, receiving_yards=30)
                    elif i == 2:
                        row.update(targets=20, receptions=15, receiving_yards=170, receiving_tds=1)
                    if i != 3:  # Missing stat row must become a known zero, not disappear.
                        stats.append(row)
                    for c in PLAYER_TARGETS:
                        team_total[c] += row[c]
                    snaps.append(dict(game_id=game_id, team=team, opponent=opponent, season=season, week=week, pfr_player_id=pfr, offense_snaps=60 if i < 3 else 0, offense_pct=1. if i < 3 else 0.))
                totals.append(team_total)
    return dict(player_stats=pd.DataFrame(stats), rosters=pd.DataFrame(rosters), snaps=pd.DataFrame(snaps), player_ids=pd.DataFrame(identities).drop_duplicates(), team_games=pd.DataFrame(teams), team_stats=pd.DataFrame(totals))


def test_player_roster_zeroes_and_reconciliation(player_sources):
    result = build_player_dataset(**player_sources)
    p = result.players
    assert not p.duplicated(["game_id", "team", "player_id"]).any()
    assert p.loc[p.week.eq(1)].player_id.eq(OTHER).all()
    assert p.loc[p.position.eq("TE"), "targets"].eq(0).all()
    assert p.loc[p.position.eq("TE"), "offense_active"].eq(0).all()
    named = p.loc[p.is_other.eq(0)]
    assert named.history_end.lt(named.gameday).all()
    first_qb = named.loc[named.week.eq(2) & named.position.eq("QB")].iloc[0]
    assert first_qb.prior_attempts_last3 == 31
    assert first_qb.attempts == 32
    assert first_qb.prior_roster_games == 1
    for target in PLAYER_TARGETS:
        np.testing.assert_array_equal(result.reconciliation[target], result.reconciliation[f"team_{target}"])


def test_current_outcomes_roster_status_cannot_change_features(player_sources):
    before = build_player_dataset(**player_sources).players
    changed = {name: frame.copy() for name, frame in player_sources.items()}
    mask = changed["player_stats"].season.eq(2025) & changed["player_stats"].week.eq(4) & changed["player_stats"].position.eq("QB")
    changed["player_stats"].loc[mask, "attempts"] += 9
    mask = changed["team_stats"].game_id.eq("2025_4_A_B")
    changed["team_stats"].loc[mask, "attempts"] += 9
    changed["rosters"].loc[changed["rosters"].season.eq(2025) & changed["rosters"].week.eq(4), "status"] = "RES"
    after = build_player_dataset(**changed).players
    compare = ["game_id", "team", "player_id", "history_end", *PLAYER_FEATURES]
    assert_frame_equal(before[compare], after[compare])
    assert_frame_equal(before.loc[before.season.lt(2025)], after.loc[after.season.lt(2025)])


def test_future_roster_addition_does_not_select_current_candidates(player_sources):
    before = build_player_dataset(**player_sources).players
    extra = player_sources["rosters"].iloc[:1].copy()
    extra = extra.assign(week=3, gsis_id="NEW", full_name="New Player", pfr_id="NEW_PFR")
    player_sources["rosters"] = pd.concat([player_sources["rosters"], extra], ignore_index=True)
    after = build_player_dataset(**player_sources).players
    assert "NEW" not in after.loc[after.game_id.eq("2021_3_A_B"), "player_id"].tolist()
    assert "NEW" in after.loc[after.game_id.eq("2021_4_A_B"), "player_id"].tolist()
    cols = ["game_id", "team", "player_id", *PLAYER_FEATURES]
    assert_frame_equal(before.loc[before.game_id.eq("2021_3_A_B"), cols].reset_index(drop=True), after.loc[after.game_id.eq("2021_3_A_B"), cols].reset_index(drop=True))


def test_player_history_prefix_and_order_invariance(player_sources):
    result = build_player_dataset(**player_sources).players
    shuffled = {k: v.sample(frac=1, random_state=43) for k, v in player_sources.items()}
    assert_frame_equal(result, build_player_dataset(**shuffled).players)
    prefix = {k: v.loc[v.season.lt(2025)] if "season" in v else (v.loc[~v.game_id.str.startswith("2025")] if "game_id" in v else v) for k, v in player_sources.items()}
    assert_frame_equal(result.loc[result.season.lt(2025)].reset_index(drop=True), build_player_dataset(**prefix).players)


@pytest.mark.parametrize("problem", ["duplicate", "wrong_opponent", "totals", "missing_snaps", "bad_id_production"])
def test_bad_player_source_data_rejected(player_sources, problem):
    if problem == "duplicate":
        player_sources["player_stats"] = pd.concat([player_sources["player_stats"], player_sources["player_stats"].iloc[:1]])
    elif problem == "wrong_opponent":
        player_sources["player_stats"].loc[0, "opponent_team"] = "C"
    elif problem == "totals":
        player_sources["team_stats"].loc[0, "attempts"] += 1
    elif problem == "missing_snaps":
        player_sources["snaps"] = player_sources["snaps"].loc[~player_sources["snaps"].game_id.eq("2021_1_A_B")]
    else:
        player_sources["player_stats"].loc[0, "player_id"] = np.nan
    with pytest.raises(ValueError):
        build_player_dataset(**player_sources)


def test_player_model_targets_are_not_predictors(player_sources):
    players = build_player_dataset(**player_sources).players
    train, forecast = players.loc[players.season.lt(2025)], players.loc[players.season.eq(2025)]
    for kind in ("trained", "recent_average"):
        model = PlayerOpportunityModel(kind).fit(train)
        shares = model.predict_shares(forecast)
        np.testing.assert_allclose(shares.groupby([forecast.game_id, forecast.team]).sum(), 1)
        no_targets = forecast.drop(columns=[*PLAYER_TARGETS, "offense_active", "offense_snaps", "offense_pct", *(f"team_{c}" for c in PLAYER_TARGETS)])
        assert_frame_equal(shares, model.predict_shares(no_targets))
        assert model.fit_end_ < forecast.gameday.min()


def test_volume_model_uses_pregame_inputs_only(model_games):
    for i, target in enumerate(VOLUME_TARGETS):
        model_games[target] = 20 + i + model_games.week
    train, forecast = model_games.loc[model_games.season.lt(2025)], model_games.loc[model_games.season.eq(2025)]
    for kind in ("ridge", "training_mean", "recent_average"):
        model = TeamVolumeModel(kind).fit(train)
        prediction = model.predict(forecast)
        assert_frame_equal(prediction, model.predict(forecast.drop(columns=list(VOLUME_TARGETS))))
        assert np.isfinite(prediction).all().all()
        assert (prediction >= 0).all().all()
