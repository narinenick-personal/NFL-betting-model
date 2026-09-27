import numpy as np
import pandas as pd
import pytest

from nfl_model.quote_backtesting import validate_quotes, backtest_quotes


@pytest.fixture
def quote_data():
    # Deliberately synthetic test contract, not a historical sportsbook record.
    quotes = pd.DataFrame([dict(quote_id="fixture-1", game_id="test_game", team="A", player_id="p1", book="TEST_FIXTURE",
                                quoted_at="2025-09-07T15:00:00Z", statistic="receptions", side="over", line=3., american_odds=150,
                                participation="all_candidates", stake=2., closing_at="2025-09-07T16:59:00Z", closing_line=3., closing_odds=-110.)])
    schedules = pd.DataFrame([dict(game_id="test_game", season=2025, gameday="2025-09-07", gametime="13:00")])
    players = pd.DataFrame([dict(game_id="test_game", team="A", player_id="p1", receptions=4, offense_active=1)])
    return quotes, schedules, players


def test_quote_price_roi_and_clv_are_explicit(quote_data):
    quotes, schedules, players = quote_data
    checked = validate_quotes(quotes, schedules, players, fit_end="2025-02-09")
    assert checked.kickoff.iloc[0] == pd.Timestamp("2025-09-07T17:00:00Z")
    def provider(game_id):
        return {"A": {"candidates": players, "samples": {"receptions": np.array([[0], [3], [4], [5]]), "offense_active": np.array([[0], [1], [1], [1]])}}}
    result, summary = backtest_quotes(checked, players, provider)
    assert result.result.iloc[0] == "win"
    assert summary["net_profit"] == 3. and summary["roi"] == 1.5
    assert result.price_clv_implied_points.iloc[0] == pytest.approx((110 / 210 - .4) * 100)
    quotes["participation"] = "offense_snap_required"
    players["offense_active"] = 0
    checked = validate_quotes(quotes, schedules, players, fit_end="2025-02-09")
    result, summary = backtest_quotes(checked, players, provider)
    assert summary["voids"] == 1 and summary["roi"] == 0
    players["offense_active"] = np.nan
    with pytest.raises(ValueError, match="Unknown actual"):
        backtest_quotes(checked, players, provider)


@pytest.mark.parametrize("field,value", [("quoted_at", "2025-09-07T17:00:00Z"), ("quoted_at", "2025-09-06T15:00:00Z"),
                                         ("quoted_at", "2025-09-07 11:00:00"), ("american_odds", 0), ("line", 3.25),
                                         ("player_id", "missing"), ("stake", -1), ("closing_at", "2025-09-07T17:01:00Z"),
                                         ("closing_line", np.nan), ("participation", "unknown")])
def test_invalid_quote_contracts_and_timestamps_rejected(quote_data, field, value):
    quotes, schedules, players = quote_data
    quotes[field] = value
    with pytest.raises(ValueError):
        validate_quotes(quotes, schedules, players, fit_end="2025-02-09")


def test_changed_line_has_no_price_only_clv(quote_data):
    quotes, schedules, players = quote_data
    quotes["closing_line"] = 4.
    checked = validate_quotes(quotes, schedules, players, fit_end="2025-02-09")
    provider = lambda _: {"A": {"candidates": players, "samples": {"receptions": np.array([[4], [5]]), "offense_active": np.ones((2, 1), dtype=int)}}}
    result, _ = backtest_quotes(checked, players, provider)
    assert result.line_clv.iloc[0] == 1
    assert np.isnan(result.price_clv_implied_points.iloc[0])
