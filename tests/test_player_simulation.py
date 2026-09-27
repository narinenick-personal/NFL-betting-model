import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from test_player_dataset import player_sources
from nfl_model.player_dataset import build_player_dataset, PLAYER_TARGETS
from nfl_model.opportunity_models import PlayerOpportunityModel, VOLUME_TARGETS
from nfl_model.player_production import PlayerProductionModel, TouchdownModel, production_calibration, fit_allocation_concentrations
from nfl_model.player_simulation import allocate_multinomial, allocate_with_capacities, simulate_team_players, ConditionalVolumeDistribution
from nfl_model.player_props import PropContract, price_prop, prop_probabilities


def test_vector_allocations_reconcile_and_respect_capacities():
    rng = np.random.default_rng(234)
    totals = np.array([0, 5, 20, 60])
    probabilities = np.array([[0, 0, 1], [.1, .8, .1], [1, 0, 0], [.2, .2, .6]])
    counts = allocate_multinomial(totals, probabilities, rng=rng)
    np.testing.assert_array_equal(counts.sum(axis=1), totals)
    assert (counts[probabilities == 0] == 0).all()
    chosen = allocate_with_capacities(totals // 2, counts, rng=rng)
    np.testing.assert_array_equal(chosen.sum(axis=1), totals // 2)
    assert (chosen <= counts).all()
    with pytest.raises(ValueError, match="capacity"):
        allocate_with_capacities(totals + 1, counts, rng=rng)


def test_production_models_and_joint_player_draws(player_sources):
    rows = build_player_dataset(**player_sources).players
    train = rows.loc[rows.season.eq(2021)]
    earlier = rows.loc[rows.season.eq(2022)].copy()
    production = PlayerProductionModel().fit(train)
    opportunity = PlayerOpportunityModel().fit(train)
    efficiencies = production.predict(earlier)
    for c in efficiencies:
        earlier[c] = efficiencies[c]
    earlier["fit_end"] = train.gameday.max()
    parameters = production_calibration(earlier)
    weights = opportunity.predict_weights(earlier)
    for c in weights:
        earlier[c] = weights[c]
    concentration = fit_allocation_concentrations(earlier)
    for value in concentration.values():
        assert .49 < value["concentration"] < 1001
    candidates = rows.loc[rows.game_id.eq("2023_3_A_B") & rows.team.eq("A")].sort_values(["is_other", "player_id"]).reset_index(drop=True)
    predicted = production.predict(candidates)
    no_targets = candidates.drop(columns=[*PLAYER_TARGETS, "offense_active", "offense_snaps", "offense_pct"])
    assert_frame_equal(predicted, production.predict(no_targets))
    weights = opportunity.predict_weights(candidates)
    weights.loc[candidates.position.eq("TE"), "active_probability"] = 0
    teams = player_sources["team_games"].merge(player_sources["team_stats"], on=["game_id", "team"]).assign(points_for=21)
    td = TouchdownModel().fit(teams.loc[teams.season.eq(2021)])
    draws = 10_000
    args = dict(target_rate=.8, allocation_parameters=concentration, production_parameters=parameters, touchdown_model=td)
    outcomes = simulate_team_players(candidates, weights, predicted, {"attempts": np.full(draws, 32), "carries": np.full(draws, 22)},
                                     np.full(draws, 21), rng=np.random.default_rng(78), **args)
    for a, b in (("completions", "receptions"), ("passing_yards", "receiving_yards"), ("passing_tds", "receiving_tds")):
        np.testing.assert_array_equal(outcomes[a].sum(axis=1), outcomes[b].sum(axis=1))
    assert (outcomes["attempts"].sum(axis=1) == 32).all()
    assert (outcomes["carries"].sum(axis=1) == 22).all()
    assert (outcomes["targets"].sum(axis=1) <= 32).all()
    assert (outcomes["receptions"] <= outcomes["targets"]).all()
    assert (outcomes["completions"] <= outcomes["attempts"]).all()
    assert ((outcomes["passing_tds"] + outcomes["rushing_tds"]).sum(axis=1) * 6 <= 21).all()
    for stat in PLAYER_TARGETS:
        assert (outcomes[stat][outcomes["offense_active"] == 0] == 0).all()
    earlier["fit_end"] = earlier.gameday
    with pytest.raises(ValueError, match="out-of-time"):
        production_calibration(earlier)


def test_conditional_volume_retains_learned_score_dependence():
    rng = np.random.default_rng(22)
    scores = rng.normal(0, 8, (200, 2))
    volume = np.column_stack([scores[:, 0] * .6 + rng.normal(0, 1, 200), *[rng.normal(0, 3, 200) for _ in range(5)]])
    errors = pd.DataFrame({"game_id": [f"g{i}" for i in range(200)], "gameday": pd.date_range("2022-01-01", periods=200),
                           "mean_fit_end": pd.Timestamp("2021-12-01"), "residual_home_score": scores[:, 0], "residual_away_score": scores[:, 1]})
    volume_rows = []
    for i, row in errors.iterrows():
        for j, target in enumerate(VOLUME_TARGETS):
            volume_rows.append(dict(game_id=row.game_id, gameday=row.gameday, fit_end=row.mean_fit_end, target=target, actual=30 + volume[i, j], predicted=30))
    model = ConditionalVolumeDistribution().fit(errors, pd.DataFrame(volume_rows), prediction_start="2023-01-01")
    high = model.sample(np.tile([40, 20], (20_000, 1)), [20, 20], np.full(6, 30), rng=np.random.default_rng(1))
    low = model.sample(np.tile([10, 20], (20_000, 1)), [20, 20], np.full(6, 30), rng=np.random.default_rng(1))
    assert high.home_attempts.mean() > low.home_attempts.mean() + 10
    assert (high.to_numpy() >= 0).all() and (high.to_numpy() % 1 == 0).all()
    errors.loc[0, "mean_fit_end"] = errors.loc[0, "gameday"]
    with pytest.raises(ValueError, match="out of time"):
        ConditionalVolumeDistribution().fit(errors, pd.DataFrame(volume_rows), prediction_start="2023-01-01")


def test_player_prop_push_void_and_price_math():
    values = np.array([0, 3, 4, 5])
    active = np.array([0, 1, 1, 1])
    contract = PropContract("receptions", 4, participation="offense_snap_required")
    result = price_prop(values, contract, 150, offense_active=active)
    assert result["p_win"] == .25 and result["p_loss"] == .25
    assert result["p_push"] == .25 and result["p_void"] == .25
    assert result["expected_net_profit_per_unit"] == .125
    under = prop_probabilities(values, PropContract("receptions", 3.5, side="under"))
    assert under["p_win"] == .5 and under["p_push"] == 0
    with pytest.raises(ValueError, match="participation"):
        prop_probabilities(values, contract)
    with pytest.raises(ValueError, match="half"):
        prop_probabilities(values, PropContract("receptions", 3.25))
