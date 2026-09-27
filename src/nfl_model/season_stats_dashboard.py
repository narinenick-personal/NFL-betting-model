"""Browse published game/team/player statistics, including the current season."""
from pathlib import Path

import pandas as pd
import streamlit as st

from nfl_model.season_stats import read_snapshot, summarize_stats


@st.cache_data(show_spinner=False, max_entries=2)
def _snapshot(path: str, modified: int):
    return read_snapshot(Path(path))


def render_season_stats(root):
    st.header("Season stats")
    st.write("Explore published game results, team totals and player game logs through the 2026 NFL season.")
    path = root / "data/season_stats/stats.zip"
    if not path.exists():
        st.info("Build the stats snapshot with: python scripts/refresh_season_stats.py")
        return
    schedule, teams, players, metadata = _snapshot(str(path), path.stat().st_mtime_ns)
    season = st.selectbox("NFL season", sorted(schedule.season.unique(), reverse=True), key="stats_season")
    st.caption(f"Snapshot built: {metadata['built_at_utc']} · Published observations, not model forecasts. Scores and statistics can update at different times.")
    with st.expander("Refresh and coverage"):
        st.code(f"python scripts/refresh_season_stats.py --seasons {season} --refresh", language="bash")
        st.caption("Run this command locally, then upload the updated data/season_stats/stats.zip to GitHub to refresh a hosted app. It preserves other seasons and the trained models. There is no background refresh.")
        sources = [{"dataset": name, **source} for name, source in metadata.get("sources", {}).items() if name.endswith(f"_{season}")]
        st.dataframe(pd.DataFrame(sources), hide_index=True)
    schedule = schedule.loc[schedule.season.eq(season)]
    phase = st.radio("Season phase", ["Regular season", "Postseason", "All"], horizontal=True, key="stats_phase")
    if phase != "All":
        schedule = schedule.loc[schedule.game_type.eq("REG") if phase == "Regular season" else schedule.game_type.ne("REG")]
    weeks = st.multiselect("Weeks", sorted(schedule.week.unique()), key=f"stats_weeks_{season}_{phase}", placeholder="All weeks")
    if weeks:
        schedule = schedule.loc[schedule.week.isin(weeks)]
    team = st.selectbox("Filter team", ["All", *sorted(set(schedule.home_team) | set(schedule.away_team))], key="stats_team")
    if team != "All":
        schedule = schedule.loc[schedule.home_team.eq(team) | schedule.away_team.eq(team)]
    team_rows = teams.loc[teams.game_id.isin(schedule.game_id)]
    player_rows = players.loc[players.game_id.isin(schedule.game_id)]
    if team != "All":
        team_rows = team_rows.loc[team_rows.team.eq(team)]
        player_rows = player_rows.loc[player_rows.team.eq(team)]
    cols = st.columns(3)
    cols[0].metric("Games with team stats", int(schedule.status.eq("Stats available").sum()))
    cols[1].metric("Team-game rows", len(team_rows))
    cols[2].metric("Player-game rows", len(player_rows))
    st.caption("Totals use the selected weeks and phase. Games without published statistics stay missing. A player row does not by itself establish sportsbook participation or availability.")
    tabs = st.tabs(["Schedule & results", "Team stats", "Player stats"])
    with tabs[0]:
        st.dataframe(schedule, hide_index=True, width="stretch")
        st.download_button("Download schedule", schedule.to_csv(index=False), f"schedule_{season}.csv", "text/csv")
        st.caption("Kickoff times are US Eastern. An NFL season can include January/February games in the following calendar year.")
    for tab, rows, is_player in ((tabs[1], team_rows, False), (tabs[2], player_rows, True)):
        with tab:
            if rows.empty:
                st.info("No published statistics match these filters yet.")
                continue
            kind = "player" if is_player else "team"
            mode = st.radio("Display", ["Totals", "Game logs"], horizontal=True, key=f"stats_{kind}_mode")
            if is_player:
                if "position" in rows:
                    position = st.selectbox("Position", ["All", *sorted(rows.position.dropna().unique())], key="stats_position")
                    if position != "All":
                        rows = rows.loc[rows.position.eq(position)]
                search = st.text_input("Find player", key="stats_search")
                if search:
                    rows = rows.loc[rows.player_display_name.str.contains(search, case=False, regex=False, na=False)]
            table = summarize_stats(rows, players=is_player) if mode == "Totals" else rows
            choices = [c for c in ("passing_yards", "rushing_yards", "receiving_yards", "receptions", "targets", "def_sacks", "points_for") if c in table]
            if choices:
                sort = st.selectbox("Sort by", choices, key=f"stats_{kind}_sort")
                table = table.sort_values(sort, ascending=False, na_position="last")
            first = [c for c in ("season", "team", "player_display_name", "position", "games_with_rows", "week", "gameday", "opponent_team") if c in table]
            table = table[[*first, *(c for c in table if c not in first)]]
            st.dataframe(table.round(2), hide_index=True, width="stretch")
            st.download_button(f"Download {kind} stats", table.to_csv(index=False), f"{kind}_stats_{season}.csv", "text/csv")
            st.caption("Efficiency ratios use summed numerators and denominators. Player totals are split by team after trades; unidentified player rows appear only in game logs.")
