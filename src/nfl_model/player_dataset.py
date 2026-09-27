"""Lagged-roster player forecasts with explicit unlisted-player reconciliation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_model.data_quality import require_columns, require_unique_keys

OFFENSIVE_POSITIONS = ("QB", "RB", "FB", "WR", "TE")
PLAYER_TARGETS = ("attempts", "carries", "targets", "receptions", "completions", "passing_yards",
                  "rushing_yards", "receiving_yards", "passing_tds", "rushing_tds", "receiving_tds")
COUNT_TARGETS = tuple(c for c in PLAYER_TARGETS if not c.endswith("yards"))
HISTORY_VALUES = (*PLAYER_TARGETS, "offense_snaps", "offense_pct", "offense_active",
                  "attempt_share", "carry_share", "target_share", "catch_rate", "yards_per_catch", "yards_per_carry")
PLAYER_FEATURES = ("position", "is_other", "prior_roster_games", *(
    f"prior_{name}_{window}" for name in HISTORY_VALUES for window in ("last3", "last5", "season")
))
OTHER = "__OTHER__"


@dataclass
class PlayerDataset:
    players: pd.DataFrame
    observations: pd.DataFrame
    reconciliation: pd.DataFrame
    id_exceptions: pd.DataFrame
    metadata: dict


def _counts_valid(frame: pd.DataFrame, columns, label: str) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError(f"{label} requires finite nonnegative integer counts")


def _historical_values(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for target, denominator, name in (("attempts", "team_attempts", "attempt_share"),
                                       ("carries", "team_carries", "carry_share"),
                                       ("targets", "team_targets", "target_share"),
                                       ("receptions", "targets", "catch_rate"),
                                       ("receiving_yards", "receptions", "yards_per_catch"),
                                       ("rushing_yards", "carries", "yards_per_carry")):
        result[name] = result[target] / result[denominator].where(result[denominator].gt(0))
    return result


def _history_snapshots(observations: pd.DataFrame, *, shift: bool) -> pd.DataFrame:
    """Only used on past observations, or with shift=True on current targets."""
    rows = observations.sort_values(["season", "team", "player_id", "gameday", "game_id"]).copy()
    keys = ["season", "team", "player_id"]
    grouped = rows.groupby(keys, sort=False)
    rows["prior_roster_games"] = grouped.cumcount() + (0 if shift else 1)
    history = _historical_values(rows)
    values = history[list(HISTORY_VALUES)]
    if shift:
        values = values.groupby([history[k] for k in keys], sort=False).shift()
    grouped_values = values.groupby([history[k] for k in keys], sort=False)
    for suffix, window in (("last3", 3), ("last5", 5), ("season", None)):
        calculated = (grouped_values.rolling(window, min_periods=1).mean() if window
                      else grouped_values.expanding(min_periods=1).mean())
        calculated = calculated.reset_index(level=keys, drop=True).reindex(rows.index)
        calculated.columns = [f"prior_{name}_{suffix}" for name in calculated.columns]
        rows = pd.concat([rows, calculated], axis=1)
    return rows


def build_player_dataset(player_stats: pd.DataFrame, rosters: pd.DataFrame, snaps: pd.DataFrame,
                         player_ids: pd.DataFrame, team_games: pd.DataFrame, team_stats: pd.DataFrame) -> PlayerDataset:
    """Forecast candidates come from each team's PREVIOUS game, within season.

    First team games contain OTHER only. Current roster/status/participation never
    decides forecast eligibility. Observations include zero-usage roster players;
    unlisted current players' outcomes reconcile into OTHER after prediction.
    """
    keys = ["game_id", "team"]
    require_unique_keys(team_games, keys, "team games")
    require_unique_keys(team_stats, keys, "team stats")
    require_columns(player_stats, ["season", "week", "season_type", "opponent_team", "position", "player_display_name", *PLAYER_TARGETS], "player stats")
    empty_identity = player_stats.loc[player_stats.player_id.isna()].copy()
    if not empty_identity[list(PLAYER_TARGETS)].eq(0).all().all():
        raise ValueError("Unidentified player statistics contain offensive production")
    player_stats = player_stats.loc[player_stats.player_id.notna()].copy()
    require_unique_keys(player_stats, [*keys, "player_id"], "player stats")
    require_columns(rosters, ["season", "week", "game_type", "team", "gsis_id", "full_name", "position", "pfr_id"], "weekly rosters")
    require_unique_keys(snaps, [*keys, "pfr_player_id"], "snap counts")
    require_columns(snaps, ["offense_snaps", "offense_pct", "season", "week", "opponent"], "snap counts")
    require_columns(player_ids, ["gsis_id", "pfr_id"], "player IDs")
    teams = team_games[[*keys, "season", "week", "game_type", "season_type", "gameday", "opponent_team", "is_home"]].copy()
    teams["gameday"] = pd.to_datetime(teams.gameday)
    require_unique_keys(teams, ["season", "week", "game_type", "team"], "scheduled roster games")
    require_unique_keys(teams, ["season", "team", "gameday"], "team game dates")
    teams = teams.sort_values(["season", "team", "gameday", "game_id"])
    totals = team_stats[[*keys, *PLAYER_TARGETS]].copy()
    _counts_valid(totals, COUNT_TARGETS, "Team statistics")
    totals = totals.rename(columns={c: f"team_{c}" for c in PLAYER_TARGETS})
    teams = teams.merge(totals, on=keys, how="left", validate="one_to_one")
    if teams.filter(like="team_").isna().any().any():
        raise ValueError("Missing team outcomes")
    stats = player_stats.copy()
    identity = stats.merge(teams[keys + ["season", "week", "season_type", "opponent_team"]], on=keys, how="left", suffixes=("", "_schedule"), validate="many_to_one")
    for column in ("season", "week", "season_type", "opponent_team"):
        if not identity[column].eq(identity[f"{column}_schedule"]).all():
            raise ValueError(f"Player stats {column} disagrees with schedules")
    _counts_valid(stats, COUNT_TARGETS, "Player statistics")
    if not np.isfinite(stats[list(PLAYER_TARGETS)].to_numpy(dtype=float)).all():
        raise ValueError("Player outcomes cannot be missing or infinite")
    if (stats.receptions > stats.targets).any() or (stats.completions > stats.attempts).any():
        raise ValueError("Completions/receptions cannot exceed attempts/targets")
    grouped_stats = stats.groupby(keys)[list(PLAYER_TARGETS)].sum()
    actual = teams.set_index(keys)
    for column in PLAYER_TARGETS:
        if not grouped_stats[column].reindex(actual.index).eq(actual[f"team_{column}"]).all():
            raise ValueError(f"Player {column} does not reconcile with all team games")

    # Cross-source identifiers are audit joins only; current biographical/status
    # fields never become predictors. Ambiguous IDs are explicitly quarantined.
    mapping = pd.concat([player_ids[["gsis_id", "pfr_id"]], rosters[["gsis_id", "pfr_id"]]], ignore_index=True).dropna().drop_duplicates()
    ambiguous = mapping.loc[mapping.duplicated("pfr_id", keep=False)].copy()
    mapping = mapping.loc[~mapping.pfr_id.isin(ambiguous.pfr_id)]
    mapped = snaps.merge(mapping, left_on="pfr_player_id", right_on="pfr_id", how="left", validate="many_to_one")
    unidentified = mapped.loc[mapped.gsis_id.isna()].copy()
    mapped = mapped.loc[mapped.gsis_id.notna()].rename(columns={"gsis_id": "player_id"})
    require_unique_keys(mapped, [*keys, "player_id"], "mapped snaps")
    snap_identity = snaps.merge(teams[keys + ["season", "week", "opponent_team"]], on=keys, how="left", suffixes=("", "_schedule"), validate="many_to_one")
    if not snap_identity.season.eq(snap_identity.season_schedule).all() or not snap_identity.week.eq(snap_identity.week_schedule).all() or not snap_identity.opponent.eq(snap_identity.opponent_team).all():
        raise ValueError("Snap identity disagrees with schedule")
    if set(map(tuple, snaps[keys].drop_duplicates().to_numpy())) != set(map(tuple, teams[keys].to_numpy())):
        raise ValueError("Snap coverage must include every completed team game")
    _counts_valid(snaps, ["offense_snaps"], "Snap observations")
    if not snaps.offense_pct.between(0, 1).all():
        raise ValueError("Snap percentages must be in [0, 1]")

    roster = rosters.loc[rosters.position.isin(OFFENSIVE_POSITIONS)].copy()
    missing_ids = roster.loc[roster.gsis_id.isna()].copy()
    roster = roster.loc[roster.gsis_id.notna()].rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    roster = roster.merge(teams[[*keys, "season", "week", "game_type"]], on=["season", "week", "game_type", "team"], how="inner", validate="many_to_one")
    require_unique_keys(roster, [*keys, "player_id"], "rostered players")
    if set(map(tuple, roster[keys].drop_duplicates().to_numpy())) != set(map(tuple, teams[keys].to_numpy())):
        raise ValueError("Offensive rosters are missing completed team games")
    columns = [*keys, "player_id", "player_name", "position"]
    observed = stats.loc[stats.position.isin(OFFENSIVE_POSITIONS) | stats[["attempts", "carries", "targets"]].sum(axis=1).gt(0), [*keys, "player_id", "player_display_name", "position"]].rename(columns={"player_display_name": "player_name"})
    observations = pd.concat([roster[columns], observed[columns]], ignore_index=True).drop_duplicates([*keys, "player_id"], keep="first")
    observations = observations.merge(teams, on=keys, how="left", validate="many_to_one")
    observations = observations.merge(stats[[*keys, "player_id", *PLAYER_TARGETS]], on=[*keys, "player_id"], how="left", validate="one_to_one")
    observations[list(PLAYER_TARGETS)] = observations[list(PLAYER_TARGETS)].fillna(0)
    observations = observations.merge(mapped[[*keys, "player_id", "offense_snaps", "offense_pct"]], on=[*keys, "player_id"], how="left", validate="one_to_one")
    known_ids = set(mapping.gsis_id) - set(ambiguous.gsis_id)
    known = observations.player_id.isin(known_ids)
    observations.loc[known, ["offense_snaps", "offense_pct"]] = observations.loc[known, ["offense_snaps", "offense_pct"]].fillna(0)
    observations["offense_active"] = np.where(observations.offense_snaps.notna(), observations.offense_snaps.gt(0).astype(float), np.nan)
    observations.loc[observations[["attempts", "carries", "targets"]].sum(axis=1).gt(0), "offense_active"] = 1.0
    observations["position"] = observations.position.where(observations.position.isin(OFFENSIVE_POSITIONS), "OTHER")
    observations["is_other"] = 0
    snapshots = _history_snapshots(observations, shift=False)
    snapshots = snapshots[[*keys, "player_id", "player_name", "gameday", *PLAYER_FEATURES]].rename(columns={"game_id": "history_game_id", "gameday": "history_end"})
    teams["history_game_id"] = teams.groupby(["season", "team"], sort=False).game_id.shift()
    named = teams.loc[teams.history_game_id.notna()].merge(snapshots, on=["history_game_id", "team"], how="left", validate="one_to_many")
    if named.player_id.isna().any() or not named.history_end.lt(named.gameday).all():
        raise ValueError("Forecast candidates need a strictly earlier roster snapshot")
    named = named.merge(stats[[*keys, "player_id", *PLAYER_TARGETS]], on=[*keys, "player_id"], how="left", validate="one_to_one")
    named[list(PLAYER_TARGETS)] = named[list(PLAYER_TARGETS)].fillna(0)
    named = named.merge(mapped[[*keys, "player_id", "offense_snaps", "offense_pct"]], on=[*keys, "player_id"], how="left", validate="one_to_one")
    known = named.player_id.isin(known_ids)
    named.loc[known, ["offense_snaps", "offense_pct"]] = named.loc[known, ["offense_snaps", "offense_pct"]].fillna(0)
    named["offense_active"] = np.where(named.offense_snaps.notna(), named.offense_snaps.gt(0).astype(float), np.nan)
    named.loc[named[["attempts", "carries", "targets"]].sum(axis=1).gt(0), "offense_active"] = 1.0

    # Every team/game gets an OTHER row, including season openers with no history.
    other = teams.copy()
    sums = named.groupby(keys)[list(PLAYER_TARGETS)].sum().reindex(pd.MultiIndex.from_frame(other[keys]), fill_value=0)
    for col in PLAYER_TARGETS:
        other[col] = other[f"team_{col}"].to_numpy() - sums[col].to_numpy()
    _counts_valid(other, COUNT_TARGETS, "Unlisted player reconciliation")
    other = other.assign(player_id=OTHER, player_name="Other / unlisted players", position="OTHER", is_other=1,
                         offense_active=1., offense_snaps=np.nan, offense_pct=np.nan)
    other["history_end"] = other.groupby(["season", "team"], sort=False).gameday.shift()
    other = _history_snapshots(other, shift=True)
    players = pd.concat([named, other], ignore_index=True).sort_values(["gameday", "game_id", "team", "player_id"]).reset_index(drop=True)
    require_unique_keys(players, [*keys, "player_id"], "player forecast dataset")
    reconciliation = players.groupby(keys)[list(PLAYER_TARGETS)].sum().join(actual[[f"team_{col}" for col in PLAYER_TARGETS]])
    for col in PLAYER_TARGETS:
        if not np.allclose(reconciliation[col], reconciliation[f"team_{col}"]):
            raise ValueError(f"Forecast candidate/OTHER {col} reconciliation failed")
    exceptions = pd.concat([
        empty_identity.assign(reason="unidentified_zero_offense_stats"),
        missing_ids.assign(reason="roster_missing_gsis"),
        ambiguous.assign(reason="ambiguous_pfr_mapping"),
        unidentified.assign(reason="snap_missing_or_ambiguous_gsis"),
    ], ignore_index=True)
    metadata = {
        "schema_version": 1, "seasons": sorted(int(s) for s in teams.season.unique()),
        "team_games": len(teams), "games": int(teams.game_id.nunique()), "player_rows": len(players),
        "named_rows": int(players.is_other.eq(0).sum()), "feature_columns": list(PLAYER_FEATURES),
        "target_columns": list(PLAYER_TARGETS), "missing_roster_id_rows": len(missing_ids),
        "unidentified_zero_offense_stats_rows": len(empty_identity),
        "ambiguous_pfr_links": len(ambiguous), "unmapped_snap_rows": len(unidentified),
        "unmapped_offense_snaps": float(unidentified.offense_snaps.sum()),
        "unknown_named_activity_rows": int(named.offense_active.isna().sum()),
        "candidate_policy": "Previous team-game offensive roster plus previous observed offensive participants; all roster statuses retained. No season carry. Season openers contain OTHER only.",
        "history_policy": "Last 3/5 observed roster games and season averages within player/team/season; includes zero-usage roster observations. A traded player's new-team history starts fresh.",
        "other_opportunity_fraction": {c: float(other[c].sum() / teams[f'team_{c}'].sum()) for c in ("attempts", "carries", "targets")},
        "limitations": [
            "Retrospective nflverse snapshots are not archived as-of feeds. Lagging cannot remove later stat or identity corrections.",
            "No current-week injury, depth-chart, active-list, or roster status is used; lagged candidates miss some new arrivals and retain some departures.",
            "Season openers have no named-player projections. Unlisted production is retained in OTHER, not discarded.",
            "Offensive participation is not equivalent to any-snap sportsbook action rules. Unknown snap identities are not silently graded as inactive.",
            "Historical player-prop lines and settlement contracts are not available in these sources.",
        ],
        "sources": ["https://nflreadpy.nflverse.com/api/load_functions/", "https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html", "https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html"],
    }
    return PlayerDataset(players, observations, reconciliation.reset_index(), exceptions, metadata)
