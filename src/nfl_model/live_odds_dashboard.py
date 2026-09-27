"""Session-local odds workspace with manual, CSV, and explicit API ingestion."""
from __future__ import annotations

from datetime import datetime, time, timedelta
import os
from uuid import uuid4

import pandas as pd
import streamlit as st

from nfl_model.live_odds import (GAME_MARKETS, LIVE_COLUMNS, PROP_MARKETS, REGIONS, TEAMS,
                                empty_quotes, fetch_odds, merge_quotes, quote_board)
from nfl_model.player_props import PROP_STATS
from nfl_model.settlement import price_comparison


def _save(incoming):
    st.session_state.live_quotes = merge_quotes(st.session_state.live_quotes, incoming)


def _manual_entry():
    market = st.selectbox("Quote market", ["moneyline", "spread", "total", "player_prop"],
                          format_func=lambda x: x.replace("_", " ").title(), key="live_market")
    with st.form("live_manual_form"):
        book = st.text_input("Sportsbook", placeholder="e.g. fanduel", key="live_book")
        cols = st.columns(2)
        away = cols[0].selectbox("Away team", list(TEAMS), format_func=TEAMS.get, index=list(TEAMS).index("BAL"), key="live_away")
        home = cols[1].selectbox("Home team", list(TEAMS), format_func=TEAMS.get, index=list(TEAMS).index("KC"), key="live_home")
        cols = st.columns(3)
        today = pd.Timestamp.now(tz="America/New_York").date()
        day = cols[0].date_input("Kickoff date", value=today + timedelta(days=1), key="live_day")
        clock = cols[1].time_input("Kickoff time", value=time(13), key="live_time")
        zone = cols[2].selectbox("Kickoff timezone", ["America/New_York", "UTC"], key="live_zone")
        side = st.selectbox("Selection", ["home", "away"] if market in ("moneyline", "spread") else ["over", "under"], key=f"live_side_{market}")
        line = None
        if market != "moneyline":
            label = "Selected team's handicap" if market == "spread" else "Line"
            line = st.number_input(label, value=-3.5 if market == "spread" else 44.5 if market == "total" else 49.5,
                                   step=.5, key=f"live_line_{market}")
            if market == "spread":
                st.caption("Enter the handicap beside the team you selected: for example, away +3.5.")
        player, statistic, participation = "", "", ""
        if market == "player_prop":
            player = st.text_input("Player name", key="live_player")
            statistic = st.selectbox("Player statistic", list(PROP_STATS), key="live_statistic")
            participation = st.selectbox("Participation rule", ["book_rules", "all_candidates", "offense_snap_required"],
                                         format_func={"book_rules": "Book rules — not verified",
                                                      "all_candidates": "Count nonparticipation as zero",
                                                      "offense_snap_required": "Void without offensive participation"}.get, key="live_participation")
        odds = st.number_input("Quote American odds", value=-110, step=1, key="live_odds")
        st.caption("Saving timestamps the quote now. Use CSV import for quotes observed earlier. Full-game pregame markets only.")
        if st.form_submit_button("Save quote"):
            try:
                kickoff = pd.Timestamp(datetime.combine(day, clock)).tz_localize(zone, ambiguous="NaT", nonexistent="NaT")
                _save(pd.DataFrame([dict(quote_id="manual_" + uuid4().hex, book=book, away_team=away,
                                        home_team=home, kickoff=kickoff, quoted_at=pd.Timestamp.now(tz="UTC"),
                                        market=market, side=side, line=line, american_odds=odds,
                                        player=player, statistic=statistic, participation=participation, source="manual")]))
                st.success("Quote saved to this session.")
            except (ValueError, TypeError) as error:
                st.error(str(error))


def _csv_import():
    st.download_button("Download live odds CSV template", ",".join(LIVE_COLUMNS) + "\n",
                       file_name="live_odds_template.csv", mime="text/csv")
    st.caption("Use a unique quote_id per observation, NFL team codes, and timestamps with Z or a UTC offset. Leave line empty for moneylines. Spread line is the selected team's handicap. Player fields stay empty for game markets.")
    with st.expander("Supported CSV values"):
        st.write("market: moneyline, spread, total, player_prop. side: home/away for game sides, over/under for totals and props. participation: book_rules, all_candidates, offense_snap_required. source is optional provenance text and is not independently verified.")
        st.write("Player statistics: " + ", ".join(PROP_STATS))
        st.caption("This template differs from Archived quote replay. It records current quotes without requiring historical game/player IDs or settled results.")
    upload = st.file_uploader("Live odds CSV", type=["csv"], key="live_csv")
    if st.button("Import quotes", disabled=upload is None, key="live_import"):
        try:
            if upload.size > 5_000_000:
                raise ValueError("Use a CSV smaller than 5 MB")
            upload.seek(0)
            rows = pd.read_csv(upload, dtype={"quote_id": str})
            if "source" not in rows:
                rows["source"] = "csv"
            before = len(st.session_state.live_quotes)
            _save(rows)
            st.success(f"Imported {len(st.session_state.live_quotes) - before:,} new quotes. Repeated identical IDs are ignored.")
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as error:
            st.error(str(error))


def _configured_key():
    key = os.environ.get("ODDS_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except FileNotFoundError:
        return ""


def _fetch(api_key, *, markets, region, event_id=None):
    try:
        now = pd.Timestamp.now(tz="UTC")
        last = st.session_state.get("live_api_attempt")
        if last is not None and (now - last).total_seconds() < 10:
            st.info("Wait 10 seconds between provider requests.")
            return
        st.session_state.live_api_attempt = now
        with st.spinner("Getting the latest available odds…"):
            rows, events, usage = fetch_odds(api_key, markets=markets, region=region, event_id=event_id)
            if len(rows):
                _save(rows)
            if event_id is None:
                st.session_state.live_api_events = events
            st.session_state.live_api_usage = usage
        st.success(f"Received {len(rows):,} pregame quotes.")
        if usage["skipped"]:
            st.info(f"Skipped {usage['skipped']} unsupported or invalid outcomes.")
    except ValueError as error:
        st.error(str(error))


def _api_connection():
    configured = _configured_key()
    personal = st.text_input("Your Odds API key (optional if configured on the server)", type="password", key="live_api_key")
    key = personal or configured
    st.caption("Uses The Odds API. Your key is sent only to that provider. Fetches run only when you click a refresh button and consume your provider quota. Server credentials share that quota across visitors.")
    with st.expander("API setup"):
        st.markdown("Get a key from [The Odds API](https://the-odds-api.com/), then enter it above or configure Streamlit secrets locally or on your host.")
        st.code('ODDS_API_KEY = "your-key-here"', language="toml")
        st.caption("Local file: .streamlit/secrets.toml. On Streamlit Cloud: App settings → Secrets. Keep the real secrets file out of GitHub. The ODDS_API_KEY environment variable also works.")
    st.write("**Connection:** " + ("Key available; refresh to verify" if key else "No key configured"))
    region = st.selectbox("Bookmaker region", list(REGIONS), key="live_region")
    markets = st.multiselect("Game markets to fetch", list(GAME_MARKETS), default=list(GAME_MARKETS),
                             format_func=GAME_MARKETS.get, key="live_api_markets")
    if st.button("Refresh NFL game odds", disabled=not key or not markets, key="live_api_games"):
        _fetch(key, markets=markets, region=region)
    events = [e for e in st.session_state.get("live_api_events", []) if pd.Timestamp(e["kickoff"]) > pd.Timestamp.now(tz="UTC")]
    if events:
        labels = {e["id"]: f"{e['away_team']} at {e['home_team']} · {pd.Timestamp(e['kickoff']).tz_convert('America/New_York'):%b %d %H:%M %Z}" for e in events}
        event_id = st.selectbox("Matchup for API player props", list(labels), format_func=labels.get, key="live_api_event")
        props = st.multiselect("Player markets to fetch", list(PROP_MARKETS),
                               default=["player_pass_yds", "player_rush_yds", "player_reception_yds"],
                               format_func=PROP_MARKETS.get, key="live_api_props")
        if st.button("Refresh selected player props", disabled=not key or not props, key="live_api_player_refresh"):
            _fetch(key, markets=props, region=region, event_id=event_id)
        st.caption("Props are requested one matchup at a time. Availability depends on your provider account and sportsbooks. Provider player names are not automatically matched to historical model IDs; participation rules remain unverified.")
    else:
        st.caption("Refresh NFL game odds first to select a matchup for player props.")
    usage = st.session_state.get("live_api_usage")
    if usage:
        st.caption(f"Last fetch: {usage['fetched_at']} · Credits remaining: {usage['remaining']} · Last request cost: {usage['last']}")


def _estimate(board):
    with st.expander("Compare a quote with your own probability estimate"):
        recent = board.loc[board.status.eq("Recent")]
        if recent.empty:
            st.info("Add a recent pregame quote to use the calculator.")
            return
        labels = {r.quote_id: f"{r.book} · {r.away_team} at {r.home_team} · {r.player or r.market} {r.statistic} {r.side} {'' if pd.isna(r.line) else r.line} ({r.american_odds:+g})" for r in recent.itertuples()}
        choice = st.selectbox("Quote to evaluate", list(labels), format_func=labels.get, key="live_estimate_quote")
        row = recent.set_index("quote_id").loc[choice]
        st.caption("These are your assumptions, not a forecast from the saved 2025 models. Win probability is conditional on the bet being decided. Check the sportsbook's settlement rules.")
        cols = st.columns(3)
        win = cols[0].number_input("Your win probability (%)", min_value=0., max_value=100., value=50., step=.5, key="live_estimate_win") / 100
        returned = cols[1].number_input("Your push or void probability (%)", min_value=0., max_value=100., value=0., step=.5, key="live_estimate_returned") / 100
        stake = cols[2].number_input("Scenario stake ($)", min_value=0., value=10., step=1., key="live_estimate_stake")
        result = price_comparison(win * (1 - returned), (1 - win) * (1 - returned), returned, row.american_odds)
        cols = st.columns(3)
        cols[0].metric("Quoted break-even", f"{result['implied_probability']:.1%}")
        edge = result["edge_vs_price_conditional"]
        cols[1].metric("Your estimated edge", "—" if pd.isna(edge) else f"{edge * 100:+.1f} pp")
        cols[2].metric("Your estimated net profit", f"${result['expected_net_profit_per_unit'] * stake:+.2f}")


def render_live_odds():
    st.header("Live odds & lines")
    st.write("Build a current pregame quote board from sportsbook prices you enter, import, or fetch.")
    st.info("Current-game model forecasts are not connected yet. The saved models cover 2025 historical games; this board shows quoted probabilities and optional estimates you supply.")
    st.caption("Quotes are kept in this browser session. Download the history before reloading or leaving; hosted restarts do not preserve it. Refresh retrieves snapshots, not a streaming feed. In-play markets are excluded.")
    if "live_quotes" not in st.session_state:
        st.session_state.live_quotes = empty_quotes()
    tabs = st.tabs(["Manual entry", "CSV import", "API connection"])
    with tabs[0]:
        _manual_entry()
    with tabs[1]:
        _csv_import()
    with tabs[2]:
        _api_connection()
    history = st.session_state.live_quotes
    st.subheader("Latest observed quotes")
    if history.empty:
        st.info("Your board is empty. Save a quote, import a CSV, or connect the API above.")
        return
    st.download_button("Download quote history", history.to_csv(index=False), file_name="live_odds_history.csv", mime="text/csv", key="live_export")
    if st.button("Clear this session's quotes", key="live_clear"):
        st.session_state.live_quotes = empty_quotes()
        st.rerun()
    stale = st.number_input("Mark quotes stale after (minutes)", min_value=1, max_value=1440, value=15, key="live_stale")
    st.button("Recheck quote ages", key="live_recheck")
    board = quote_board(history, stale_minutes=stale)
    cols = st.columns(3)
    cols[0].metric("Saved snapshots", f"{len(history):,}")
    cols[1].metric("Recent selections", f"{board.status.eq('Recent').sum():,}")
    cols[2].metric("Stale / started", f"{board.status.ne('Recent').sum():,}")
    board["matchup"] = board.away_team + " at " + board.home_team + " · " + board.kickoff.dt.strftime("%Y-%m-%d %H:%M UTC")
    match = st.selectbox("Filter matchup", ["All", *board.matchup.unique()], key="live_filter_game")
    market = st.selectbox("Filter market", ["All", *board.market.unique()], key="live_filter_market")
    visible = board.loc[(board.matchup.eq(match) | (match == "All")) & (board.market.eq(market) | (market == "All"))].copy()
    visible["break_even_pct"] = 100 * visible.implied_probability
    columns = ["book", "matchup", "market", "player", "statistic", "side", "line", "american_odds", "break_even_pct",
               "quoted_at", "age_minutes", "status", "best_same_line", "participation", "source"]
    st.dataframe(visible[columns].round({"break_even_pct": 2, "age_minutes": 1}), hide_index=True, width="stretch")
    st.caption("Best same line marks the highest payout among recent quotes at two or more books for the same selection, line and stated participation rule. Unverified prop rules are excluded. Only the latest main line per book/selection is shown; earlier observations remain in your export. A recent quote may still have moved or been withdrawn.")
    _estimate(visible)
