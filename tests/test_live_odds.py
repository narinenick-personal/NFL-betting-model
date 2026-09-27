from copy import deepcopy
from io import BytesIO, StringIO
import json
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from nfl_model import live_odds
from nfl_model.live_odds import (OddsAPIError, empty_quotes, fetch_odds, merge_quotes,
                                normalize_api_quotes, quote_board, validate_live_quotes)

NOW = pd.Timestamp("2026-09-26T12:00:00Z")


def quote(**overrides):
    return dict(quote_id="one", book="book_a", away_team="BAL", home_team="KC",
                kickoff="2026-09-27T17:00:00Z", quoted_at="2026-09-26T11:59:00Z",
                market="spread", side="away", line=3.5, american_odds=-110,
                player="", statistic="", participation="", source="manual") | overrides


def test_csv_roundtrip_and_idempotent_import_preserves_timestamp():
    quotes = validate_live_quotes(pd.DataFrame([quote(away_team="Baltimore Ravens", home_team="kc", book="Book_A")]), now=NOW)
    restored = validate_live_quotes(pd.read_csv(StringIO(quotes.to_csv(index=False))), now=NOW)
    assert_frame_equal(quotes, restored)
    assert_frame_equal(quotes, merge_quotes(quotes, restored, now=NOW))
    assert quotes.iloc[0].line == 3.5  # Away handicap, not the grid's home handicap.
    assert quotes.iloc[0].quoted_at == pd.Timestamp("2026-09-26T11:59:00Z")
    with pytest.raises(ValueError, match="different values"):
        merge_quotes(quotes, pd.DataFrame([quote(american_odds=110)]), now=NOW)


@pytest.mark.parametrize("change,match", [
    ({"american_odds": 0}, "American odds"),
    ({"american_odds": float("inf")}, "American odds"),
    ({"line": 3.25}, "half-points"),
    ({"quoted_at": "2026-09-26T11:59:00"}, "timezone"),
    ({"quoted_at": "2026-09-27T12:00:00Z"}, "future"),
    ({"kickoff": "2026-09-26T11:00:00Z"}, "pregame"),
    ({"home_team": "BAL"}, "must differ"),
    ({"home_team": "FAKE"}, "recognized NFL"),
    ({"market": "moneyline", "line": 0}, "no line"),
    ({"market": "total", "side": "over", "line": -1}, "nonnegative"),
    ({"player": "Someone"}, "empty for game markets"),
    ({"market": "player_prop", "side": "over", "statistic": "receptions"}, "player name"),
])
def test_invalid_quotes_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        validate_live_quotes(pd.DataFrame([quote(**change)]), now=NOW)


def test_board_line_movement_staleness_best_price_and_unknown_prop_contracts():
    rows = [quote(quote_id="old", quoted_at="2026-09-26T11:55:00Z", line=4.5, american_odds=200),
            quote(quote_id="new", line=3.5, american_odds=-120),
            quote(quote_id="other", book="book_b", line=3.5, american_odds=-110),
            quote(quote_id="stale", book="stale_book", quoted_at="2026-09-26T10:00:00Z", american_odds=500),
            quote(quote_id="different_line", book="book_c", line=2.5, american_odds=200),
            quote(quote_id="past", kickoff="2026-09-26T11:59:30Z"),
            quote(quote_id="prop1", market="player_prop", side="over", player="A Player", statistic="receptions", participation="book_rules"),
            quote(quote_id="prop2", book="book_b", market="player_prop", side="over", player="A Player", statistic="receptions", participation="book_rules", american_odds=100)]
    board = quote_board(validate_live_quotes(pd.DataFrame(rows), now=NOW), now=NOW).set_index("quote_id")
    assert "old" not in board.index
    assert board.loc["other", "best_same_line"]
    assert board.best_same_line.sum() == 1
    assert board.loc["stale", "status"] == "Stale"
    assert board.loc["past", "status"] == "Started / past"
    assert board.loc["new", "implied_probability"] == pytest.approx(120 / 220)


def test_moneyline_best_price_and_known_prop_rules():
    rows = [quote(market="moneyline", line=np.nan), quote(quote_id="ml2", book="book_b", market="moneyline", line=np.nan, american_odds=120)]
    for key, rule, book, price in [("p1", "all_candidates", "book_a", -110), ("p2", "all_candidates", "book_b", 100), ("p3", "offense_snap_required", "book_c", 200)]:
        rows.append(quote(quote_id=key, book=book, market="player_prop", side="over", player="A Player", statistic="receptions", participation=rule, american_odds=price))
    board = quote_board(validate_live_quotes(pd.DataFrame(rows), now=NOW), now=NOW)
    assert set(board.loc[board.best_same_line, "quote_id"]) == {"ml2", "p2"}


def payload():
    return [{"id": "event123", "sport_key": "americanfootball_nfl", "commence_time": "2099-09-27T17:00:00Z",
             "home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens", "bookmakers": [
                 {"key": "fanduel", "last_update": "2026-09-26T11:59:00Z", "markets": [
                     {"key": "h2h", "outcomes": [{"name": "Kansas City Chiefs", "price": -150}, {"name": "Baltimore Ravens", "price": 130}]},
                     {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "price": -110, "point": -3.5}, {"name": "Baltimore Ravens", "price": -110, "point": 3.5}]},
                     {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 44.5}]},
                     {"key": "player_pass_yds", "last_update": "2026-09-26T11:58:00Z", "outcomes": [{"name": "Over", "description": "A Quarterback", "price": -115, "point": 225.5}]}]}]}]


def test_api_normalization_dedup_signs_timestamps_missing_rules_and_inplay():
    data = payload()
    past = deepcopy(data[0])
    past["commence_time"] = "2026-09-26T11:00:00Z"
    data.append(past)
    rows, events, skipped = normalize_api_quotes(data, now=NOW)
    assert len(rows) == 6 and len(events) == 1 and skipped == 0
    spread = rows.loc[rows.market.eq("spread")].set_index("side")
    assert spread.loc["away", "line"] == 3.5 and spread.loc["home", "line"] == -3.5
    prop = rows.loc[rows.market.eq("player_prop")].iloc[0]
    assert prop.participation == "book_rules" and prop.statistic == "passing_yards"
    assert prop.quoted_at == pd.Timestamp("2026-09-26T11:58:00Z")
    again, _, _ = normalize_api_quotes(data[0], now=NOW)
    assert_frame_equal(rows, merge_quotes(rows, again, now=NOW))
    data[0]["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = 0
    rows, _, skipped = normalize_api_quotes(data, now=NOW)
    assert len(rows) == 5 and skipped == 1


def test_api_transport_parameters_and_no_unrequested_network(monkeypatch):
    seen = []

    def transport(url, timeout):
        seen.append((urlparse(url), timeout))
        response = BytesIO(json.dumps(payload()).encode())
        response.headers = {"x-requests-remaining": "497", "x-requests-used": "3", "x-requests-last": "3"}
        return response

    monkeypatch.setattr(live_odds, "urlopen", transport)
    rows, events, usage = fetch_odds("test-secret")
    assert len(rows) == 6 and usage["remaining"] == "497"
    url, timeout = seen[0]
    params = parse_qs(url.query)
    assert url.netloc == "api.the-odds-api.com" and url.path.endswith("/odds")
    assert params["oddsFormat"] == ["american"] and params["markets"] == ["h2h,spreads,totals"]
    assert timeout == 15 and len(seen) == 1
    fetch_odds("test-secret", event_id=events[0]["id"], markets=["player_pass_yds"])
    assert seen[-1][0].path.endswith("/events/event123/odds")
    with pytest.raises(OddsAPIError):
        fetch_odds("test-secret", markets=["player_pass_yds"])
    with pytest.raises(OddsAPIError):
        fetch_odds("test-secret", event_id="../bad")
    assert len(seen) == 2


@pytest.mark.parametrize("failure", [HTTPError("https://example/?apiKey=test-secret", 401, "test-secret", {}, None),
                                     HTTPError("https://example/?apiKey=test-secret", 429, "test-secret", {}, None),
                                     URLError("test-secret"), TimeoutError("test-secret")])
def test_api_failure_does_not_disclose_key(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(live_odds, "urlopen", fail)
    with pytest.raises(OddsAPIError) as error:
        fetch_odds("test-secret")
    assert "test-secret" not in str(error.value)


def test_live_view_works_without_model_artifacts_and_preserves_session(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    app = AppTest.from_string(f"from pathlib import Path\nfrom nfl_model.dashboard import render_dashboard\nrender_dashboard(Path({str(tmp_path)!r}))", default_timeout=20).run()
    app.radio(key="view").set_value("Live odds & lines").run()
    assert not app.exception
    assert app.button(key="live_api_games").disabled
    assert "HISTORICAL REPLAY" not in " ".join(x.value for x in app.caption)
    app.button(key="FormSubmitter:live_manual_form-Save quote").click().run()
    assert len(app.error) == 1 and "sportsbook" in app.error[0].value
    app.text_input(key="live_book").set_value("Test book")
    app.button(key="FormSubmitter:live_manual_form-Save quote").click().run()
    assert not app.exception and not app.error
    assert len(app.session_state["live_quotes"]) == 1
    assert app.metric[0].value == "1"
    app.radio(key="view").set_value("Game markets").run()
    app.radio(key="view").set_value("Live odds & lines").run()
    assert not app.exception and len(app.session_state["live_quotes"]) == 1
    app.number_input(key="live_estimate_win").set_value(60.).run()
    assert any(x.label == "Your estimated net profit" and x.value == "$+1.45" for x in app.metric)
    app.button(key="live_clear").click().run()
    assert not app.exception and app.session_state["live_quotes"].empty


def test_api_ui_click_fetches_once_and_retains_history_on_error(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-secret")
    calls = []

    def fetch(*args, **kwargs):
        calls.append(kwargs)
        rows, events, skipped = normalize_api_quotes(payload())
        return rows, events, {"remaining": "497", "used": "3", "last": "3", "skipped": skipped, "fetched_at": NOW.isoformat()}

    monkeypatch.setattr("nfl_model.live_odds_dashboard.fetch_odds", fetch)
    app = AppTest.from_string("from nfl_model.live_odds_dashboard import render_live_odds\nrender_live_odds()", default_timeout=20).run()
    assert not app.exception and not calls
    app.button(key="live_api_games").click().run()
    assert not app.exception and len(calls) == 1
    assert len(app.session_state["live_quotes"]) == 6
    app.run()
    assert len(calls) == 1
    def fail(*args, **kwargs):
        raise OddsAPIError("API quota or rate limit reached")
    monkeypatch.setattr("nfl_model.live_odds_dashboard.fetch_odds", fail)
    app.session_state["live_api_attempt"] = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=20)
    app.button(key="live_api_games").click().run()
    assert not app.exception and "quota" in app.error[0].value
    assert len(app.session_state["live_quotes"]) == 6
