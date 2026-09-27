"""Local historical research dashboard backed by verified model artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from nfl_model.player_props import PROP_STATS, PropContract, price_prop
from nfl_model.settlement import price_comparison
from nfl_model.quote_backtesting import QUOTE_COLUMNS, validate_quotes, backtest_quotes
from nfl_model.live_odds_dashboard import render_live_odds
from nfl_model.season_stats_dashboard import render_season_stats


@st.cache_data(show_spinner=False)
def _read_frame(path: str, modified: int):
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def _read_models(path: str, modified: int):
    # Only the project's own generated files are accepted by this UI.
    return joblib.load(path)


def _frame(root, relative):
    path = root / relative
    return _read_frame(str(path), path.stat().st_mtime_ns)


def _models(root, relative):
    path = root / relative
    return _read_models(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False, max_entries=2)
def _simulate_game(root_path: str, game_id: str, model_version: int, player_version: int, game_version: int):
    root = Path(root_path)
    model = _models(root, "artifacts/players/models.joblib")["football_players"]
    games = _frame(root, "data/processed/game_modeling_dataset.parquet")
    players = _frame(root, "data/processed/player_modeling_dataset.parquet")
    return next(model.simulate(games.loc[games.game_id.eq(game_id)], players.loc[players.game_id.eq(game_id)], draws=10_000))


def _game_selector(games, *, key, allowed=None):
    options = games.loc[games.season.eq(2025)].sort_values(["gameday", "game_id"])
    if allowed is not None:
        options = options.loc[options.game_id.isin(allowed)]
    if options.empty:
        st.info("No historical games are available for this view.")
        return None
    labels = {r.game_id: f"{pd.Timestamp(r.gameday):%b %d} · {r.away_team} at {r.home_team} · Week {r.week}" for r in options.itertuples()}
    game_id = st.selectbox("Historical matchup", options.game_id.tolist(), format_func=labels.get, key=key)
    return options.loc[options.game_id.eq(game_id)].iloc[0]


def _percentage(value):
    return f"{value:.1%}" if pd.notna(value) else "—"


def _price_metrics(comparison):
    columns = st.columns(4)
    columns[0].metric("Model win probability", _percentage(comparison["model_probability_conditional"]))
    columns[1].metric("Quoted break-even", _percentage(comparison["implied_probability"]))
    columns[2].metric("Probability edge", f"{100 * comparison['edge_vs_price_conditional']:+.1f} pp" if pd.notna(comparison["edge_vs_price_conditional"]) else "—")
    columns[3].metric("Expected net per $1", f"${comparison['expected_net_profit_per_unit']:+.3f}")
    st.caption("Win probability and edge condition on a decided bet. Expected net profit includes returned stakes on pushes and voids.")


def _game_markets(root, games):
    st.header("Game markets")
    st.write("Replay a historical pregame forecast and compare it with the stored market snapshot.")
    row = _game_selector(games, key="market_game")
    if row is None:
        return
    model_name = st.selectbox("Score model", ["ridge_football", "ridge_market", "market_lines"],
                              format_func={"ridge_football": "Football model", "ridge_market": "Market-informed model", "market_lines": "Market-line distribution"}.get, key="score_model")
    model = _models(root, "artifacts/markets/models.joblib")[model_name]
    _, grid = next(model.grids(games.loc[games.game_id.eq(row.game_id)]))
    home_mean = float(np.dot(grid.home, grid.probabilities))
    away_mean = float(np.dot(grid.away, grid.probabilities))
    cols = st.columns(3)
    cols[0].metric(f"{row.away_team} projected score", f"{away_mean:.1f}")
    cols[1].metric(f"{row.home_team} projected score", f"{home_mean:.1f}")
    cols[2].metric("Projected total", f"{home_mean + away_mean:.1f}")
    market = st.selectbox("Market", ["moneyline", "spread", "total"], key="market")
    sides = ["over", "under"] if market == "total" else ["home", "away"]
    side = st.selectbox("Side", sides, key=f"side_{market}")
    opposite = {"home": "away", "away": "home", "over": "under", "under": "over"}[side]
    line = None
    if market == "spread":
        value = -float(row.market_spread_line) if pd.notna(row.market_spread_line) else 0.
        line = st.number_input("Home handicap", value=value, step=.5, key=f"spread_{row.game_id}")
        st.caption(f"Home handicap applies to {row.home_team}; {row.away_team} receives the opposite handicap.")
    elif market == "total":
        value = float(row.market_total_line) if pd.notna(row.market_total_line) else 44.5
        line = st.number_input("Total line", value=value, min_value=0., step=.5, key=f"total_{row.game_id}")
    price_column = f"market_{side}_moneyline" if market == "moneyline" else (f"market_{side}_spread_odds" if market == "spread" else f"market_{side}_odds")
    opposite_column = f"market_{opposite}_moneyline" if market == "moneyline" else (f"market_{opposite}_spread_odds" if market == "spread" else f"market_{opposite}_odds")
    stored = row.get(price_column, np.nan)
    odds = st.number_input("American odds", value=int(stored) if pd.notna(stored) else -110, step=1, key=f"odds_{row.game_id}_{market}_{side}")
    st.caption("Odds and lines are prefilled from untimestamped historical snapshots. Edited values are your quote scenario; these are not live prices.")
    try:
        probabilities = grid.market_probabilities(market=market, side=side, line=line)
        # Do not combine an edited quote/line with an incompatible opposite snapshot.
        unchanged_line = line is None or (market == "spread" and pd.notna(row.market_spread_line) and line == -row.market_spread_line) or (market == "total" and line == row.market_total_line)
        other = row.get(opposite_column, np.nan)
        comparison = price_comparison(**probabilities, odds=odds, opposite_odds=float(other) if unchanged_line and pd.notna(stored) and odds == stored and pd.notna(other) else None)
        _price_metrics(comparison)
        st.caption(f"Push probability: {probabilities['p_push']:.2%} · Full game including overtime · Moneyline ties return the stake")
        values = grid.home + grid.away if market == "total" else grid.home - grid.away
        distribution = pd.DataFrame({"Outcome": values, "Probability": grid.probabilities}).groupby("Outcome").Probability.sum()
        st.bar_chart(distribution.loc[distribution.gt(.0001)], x_label="Game total" if market == "total" else "Home score minus away score", y_label="Probability")
    except ValueError as error:
        st.error(str(error))


def _player_props(root, games):
    st.header("Player props")
    st.write("Explore 10,000 joint matchup simulations and enter a player-stat quote scenario.")
    if not (root / "artifacts/players/models.joblib").exists():
        st.info("Player simulation artifacts are not ready yet. Run scripts/validate_player_simulations.py after training opportunities.")
        return
    players = _frame(root, "data/processed/player_modeling_dataset.parquet")
    allowed = players.loc[players.is_other.eq(0), "game_id"].unique()
    row = _game_selector(games, key="player_game", allowed=allowed)
    if row is None:
        return
    team = st.selectbox("Team", [row.away_team, row.home_team], key="prop_team")
    with st.spinner("Simulating the matchup…"):
        _, scores, volume, records = _simulate_game(str(root), row.game_id,
                                                   (root / "artifacts/players/models.joblib").stat().st_mtime_ns,
                                                   (root / "data/processed/player_modeling_dataset.parquet").stat().st_mtime_ns,
                                                   (root / "data/processed/game_modeling_dataset.parquet").stat().st_mtime_ns)
    record = records[team]
    candidates, arrays = record["candidates"], record["samples"]
    named = candidates.loc[candidates.is_other.eq(0)].copy()
    if named.empty:
        st.info("This team has no earlier roster snapshot for named-player forecasts.")
        return
    named["prior_usage"] = named.prior_targets_last5.fillna(0) + named.prior_carries_last5.fillna(0) + named.prior_attempts_last5.fillna(0)
    named = named.sort_values(["prior_usage", "player_id"], ascending=[False, True])
    labels = {r.player_id: f"{r.player_name} · {r.position}" for r in named.itertuples()}
    player_id = st.selectbox("Player", named.player_id.tolist(), format_func=labels.get, key=f"player_{row.game_id}_{team}")
    i = int(candidates.index[candidates.player_id.eq(player_id)][0])
    position = candidates.loc[i, "position"]
    default = "passing_yards" if position == "QB" else "rushing_yards" if position in ("RB", "FB") else "receiving_yards"
    stat = st.selectbox("Statistic", list(PROP_STATS), index=PROP_STATS.index(default), format_func=lambda s: s.replace("_", " ").title(), key=f"stat_{position}")
    defaults = {"attempts": 24.5, "carries": 9.5, "targets": 4.5, "receptions": 3.5, "passing_yards": 199.5, "rushing_yards": 39.5, "receiving_yards": 39.5}
    line = st.number_input("Player line", value=defaults.get(stat, .5), step=.5, key=f"prop_line_{stat}")
    side = st.selectbox("Over / under", ["over", "under"], key="prop_side")
    odds = st.number_input("Your American odds", value=-110, step=1, key="prop_odds")
    rule = st.selectbox("Participation contract", ["all_candidates", "offense_snap_required"],
                        format_func={"all_candidates": "Count nonparticipation as zero", "offense_snap_required": "Void without offensive participation"}.get, key="prop_participation")
    st.caption("These inputs are your scenario, not retrieved sportsbook quotes. Offensive participation does not establish every book's any-snap settlement rule.")
    contract = PropContract(stat, line, side, rule)
    values, active = arrays[stat][:, i], arrays["offense_active"][:, i]
    try:
        comparison = price_prop(values, contract, odds, offense_active=active)
        _price_metrics(comparison)
        selected = values[active.astype(bool)] if rule == "offense_snap_required" else values
        st.caption(f"10,000 draws · Push {comparison['p_push']:.1%} · Void {comparison['p_void']:.1%} · Simulated offensive participation {active.mean():.1%}")
        if rule == "offense_snap_required" and comparison["action_draws"] < 100:
            st.info("Few simulations satisfy this participation contract, so its conditional estimate is imprecise.")
        if len(selected):
            lo, hi = np.quantile(selected, [.1, .9])
            st.write(f"**Mean: {selected.mean():.1f}** · Middle 80%: {lo:.0f}–{hi:.0f}")
            histogram = pd.Series(selected).value_counts(normalize=True).sort_index().rename("Probability")
            st.bar_chart(histogram, x_label=stat.replace("_", " ").title(), y_label="Probability")
        projection = candidates[["player_name", "position", "is_other"]].copy()
        for name in ("attempts", "carries", "targets", "receptions", "passing_yards", "rushing_yards", "receiving_yards"):
            projection[name] = arrays[name].mean(axis=0)
        st.subheader("Team player projections")
        st.dataframe(projection.round(2), hide_index=True, width="stretch")
        st.download_button("Download player projections", projection.to_csv(index=False), file_name=f"{row.game_id}_{team}_projections.csv", mime="text/csv")
        st.caption("Other / unlisted players retain production outside the lagged candidate roster. Player totals reconcile with simulated team opportunities.")
    except ValueError as error:
        st.error(str(error))


def _validation(root):
    st.header("Model validation")
    season = st.selectbox("Evaluation season", [2023, 2024, 2025], index=2, key="validation_season")
    st.caption("2025 is a previously inspected retrospective audit. Fits and calibration use earlier seasons; these results are not a fresh prospective test.")
    tabs = st.tabs(["Game probabilities", "Opportunities", "Player production"])
    with tabs[0]:
        rows = _frame(root, "artifacts/markets/metrics.parquet")
        st.write("Lower Brier score and log loss are better. Binary scores exclude pushes.")
        st.dataframe(rows.loc[rows.season.eq(season), ["model", "market", "games", "binary_brier", "binary_log_loss", "observed_push_rate", "mean_predicted_push_rate"]], hide_index=True, width="stretch")
    with tabs[1]:
        if (root / "artifacts/opportunities/player_metrics.parquet").exists():
            players = _frame(root, "artifacts/opportunities/player_metrics.parquet")
            st.write("Players with positive prior usage; cohort selection does not use the current result.")
            st.dataframe(players.loc[players.season.eq(season) & players.cohort.eq("prior_usage")], hide_index=True, width="stretch")
            st.subheader("Team volume")
            volume = _frame(root, "artifacts/opportunities/volume_metrics.parquet")
            st.dataframe(volume.loc[volume.season.eq(season)], hide_index=True, width="stretch")
    with tabs[2]:
        if (root / "artifacts/players/metrics.parquet").exists():
            metrics = _frame(root, "artifacts/players/metrics.parquet")
            statistic = st.selectbox("Production statistic", sorted(metrics.statistic.unique()), key="validation_statistic")
            st.dataframe(metrics.loc[metrics.season.eq(season) & metrics.statistic.eq(statistic)], hide_index=True, width="stretch")
            st.caption("Probability scores use fixed research thresholds, not historical player-prop lines. Discrete outcomes can have interval coverage above the nominal 80%.")
        else:
            st.info("The player simulation validation run has not finished yet.")


def _audit(root):
    st.header("Data audit")
    path = root / "data/processed/player_modeling_manifest.json"
    if not path.exists():
        st.info("Build the player dataset to view coverage.")
        return
    meta = json.loads(path.read_text())
    cols = st.columns(3)
    cols[0].metric("Historical games", f"{meta['games']:,}")
    cols[1].metric("Named-player forecast rows", f"{meta['named_rows']:,}")
    cols[2].metric("Unknown participation rows", f"{meta['unknown_named_activity_rows']:,}")
    st.write(meta["candidate_policy"])
    st.write(meta["history_policy"])
    st.subheader("Production assigned to other / unlisted players")
    st.dataframe(pd.DataFrame({"Opportunity": meta["other_opportunity_fraction"].keys(), "Fraction": meta["other_opportunity_fraction"].values()}), hide_index=True)
    st.caption("Includes season openers, which have no previous same-season roster snapshot.")
    with st.expander("Identity exceptions"):
        exceptions = _frame(root, "data/processed/player_id_exceptions.parquet")
        columns = [c for c in ("reason", "season", "week", "team", "player_id", "gsis_id", "full_name", "pfr_id", "pfr_player_id") if c in exceptions]
        st.dataframe(exceptions[columns], hide_index=True, width="stretch")
    for item in meta["limitations"]:
        st.caption(item)


def _quote_replay(root):
    st.header("Archived quote replay")
    st.write("Import game-day 2025 player-prop quotes to price and settle the supplied cohort. Optional closing snapshots support line and price comparisons.")
    st.download_button("Download empty quote template", ",".join(QUOTE_COLUMNS) + "\n", file_name="player_quotes.csv", mime="text/csv")
    with st.expander("CSV format and interpretation"):
        st.write("Use NFL game/player IDs, timezone-aware timestamps, whole or half lines, American odds, and a positive stake. Participation is `all_candidates` or `offense_snap_required`. Closing timestamp, line and odds must be supplied together.")
        st.caption("This replays every supplied quote without choosing bets from their results. ROI uses all staked units, including stakes later returned. Positive line CLV means a more favorable entry threshold; price CLV compares raw implied probabilities only when the line is unchanged. Supplied closing snapshots are not independently verified as the last available quote.")
        st.caption("The feature sources are retrospective snapshots, not archived historical vintages. No source files are uploaded to an external odds service.")
    uploaded = st.file_uploader("Archived quotes CSV", type=["csv"], key="quote_csv")
    if uploaded is None:
        return
    if not (root / "artifacts/players/models.joblib").exists():
        st.info("Player simulation artifacts are required to replay quotes.")
        return
    try:
        raw = pd.read_csv(uploaded)
        if len(raw) > 500:
            raise ValueError("This interactive view accepts up to 500 quotes per file")
        players = _frame(root, "data/processed/player_modeling_dataset.parquet")
        schedules = _frame(root, "data/raw/schedules_2021_2022_2023_2024_2025.parquet")
        model = _models(root, "artifacts/players/models.joblib")["football_players"]
        quotes = validate_quotes(raw, schedules, players, fit_end=model.fit_end)
        with st.spinner("Pricing and settling the supplied quotes…"):
            def provider(game_id):
                return _simulate_game(str(root), game_id, (root / "artifacts/players/models.joblib").stat().st_mtime_ns,
                                      (root / "data/processed/player_modeling_dataset.parquet").stat().st_mtime_ns,
                                      (root / "data/processed/game_modeling_dataset.parquet").stat().st_mtime_ns)[3]
            result, summary = backtest_quotes(quotes, players, provider)
        columns = st.columns(4)
        columns[0].metric("Supplied quotes", summary["quotes"])
        columns[1].metric("Net profit", f"{summary['net_profit']:+.2f} units")
        columns[2].metric("Cohort ROI", _percentage(summary["roi"]))
        columns[3].metric("Closing snapshots", summary["closing_quotes"])
        st.caption("This is the return of the uploaded cohort, not an independently selected or validated betting strategy.")
        st.dataframe(result, hide_index=True, width="stretch")
        st.download_button("Download quote results", result.to_csv(index=False), file_name="quote_replay_results.csv", mime="text/csv")
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(str(error))


def render_dashboard(root: Path):
    st.set_page_config(page_title="NFL Lab", page_icon="🏈", layout="wide")
    st.title("NFL Lab")
    view = st.sidebar.radio("Explore", ["Game markets", "Player props", "Season stats", "Live odds & lines", "Archived quote replay", "Model validation", "Data audit"], key="view")
    st.sidebar.caption("Historical model research plus a current odds workspace. Live quotes support manual entry, CSV import and an optional API connection.")
    if view == "Live odds & lines":
        render_live_odds()
        return
    if view == "Season stats":
        render_season_stats(root)
        return
    st.caption("2025 HISTORICAL REPLAY · Models fitted through the 2024 season · 10,000 player simulations per matchup")
    if not (root / "data/processed/game_modeling_dataset.parquet").exists() or not (root / "artifacts/markets/models.joblib").exists():
        st.info("Build the modeling datasets and market artifacts using the commands in README.md to open the dashboard.")
        return
    games = _frame(root, "data/processed/game_modeling_dataset.parquet")
    if view == "Game markets":
        _game_markets(root, games)
    elif view == "Player props":
        _player_props(root, games)
    elif view == "Model validation":
        _validation(root)
    elif view == "Archived quote replay":
        _quote_replay(root)
    else:
        _audit(root)
