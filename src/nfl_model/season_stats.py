"""Observed season statistics, stored separately from historical model inputs."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd

from nfl_model.data_quality import require_columns, require_unique_keys
from nfl_model.team_games import GAME_TYPES

STAT_COLUMNS = ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
                "sacks_suffered", "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
                "receiving_yards", "receiving_tds", "fumbles_lost_total", "passing_epa", "rushing_epa",
                "def_sacks", "def_interceptions", "def_tackles_solo", "penalties", "penalty_yards",
                "fg_made", "fg_att", "fantasy_points", "fantasy_points_ppr"]
SCHEDULE_COLUMNS = ["game_id", "season", "week", "game_type", "gameday", "gametime",
                    "away_team", "away_score", "home_team", "home_score"]


def prepare_season_stats(schedules, teams, players):
    """Validate identity joins; upcoming games never acquire invented zero stats."""
    require_columns(schedules, SCHEDULE_COLUMNS, "season schedules")
    schedules = schedules.loc[schedules.game_type.isin(GAME_TYPES), SCHEDULE_COLUMNS].copy()
    require_unique_keys(schedules, ["game_id"], "season schedules")
    if schedules.empty:
        raise ValueError("No regular/postseason schedules are available for the requested seasons")
    for column in ("season", "week"):
        values = pd.to_numeric(schedules[column], errors="raise")
        if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
            raise ValueError(f"Invalid schedule {column}")
        schedules[column] = values.astype(int)
    schedules["gameday"] = pd.to_datetime(schedules.gameday, errors="raise")
    if schedules[["gameday", "home_team", "away_team"]].isna().any().any() or schedules.home_team.eq(schedules.away_team).any():
        raise ValueError("Schedules require a date and two distinct teams")
    for name in ("home_score", "away_score"):
        schedules[name] = pd.to_numeric(schedules[name], errors="raise")
        values = schedules[name].dropna()
        if not np.isfinite(values).all() or values.lt(0).any() or (values % 1 != 0).any():
            raise ValueError("Invalid schedule scores")
    skeleton = []
    for side, other in (("home", "away"), ("away", "home")):
        frame = schedules[["game_id", "season", "week", "gameday"]].copy()
        frame["team"] = schedules[f"{side}_team"]
        frame["opponent_team"] = schedules[f"{other}_team"]
        frame["season_type"] = np.where(schedules.game_type.eq("REG"), "REG", "POST")
        frame["points_for"] = schedules[f"{side}_score"]
        frame["points_against"] = schedules[f"{other}_score"]
        skeleton.append(frame)
    skeleton = pd.concat(skeleton, ignore_index=True)
    output = []
    for kind, raw in (("team", teams), ("player", players)):
        identities = ["game_id", "team", "opponent_team", "season", "week", "season_type"]
        require_columns(raw, identities + (["player_id"] if kind == "player" else []), f"{kind} season stats")
        raw = raw.loc[raw.season_type.isin(["REG", "POST"])].copy()
        keys = ["game_id", "team"] + (["player_id"] if kind == "player" else [])
        require_unique_keys(raw.loc[raw.player_id.notna()] if kind == "player" else raw, keys, f"{kind} season stats")
        # Preserve upstream unknown player IDs in logs, but never merge their identities.
        names = [c for c in ("player_id", "player_display_name", "position") if c in raw] if kind == "player" else []
        values = [c for c in STAT_COLUMNS if c in raw]
        selected = raw[[*identities, *names, *values]].copy()
        for name in values:
            selected[name] = pd.to_numeric(selected[name], errors="raise")
            if np.isinf(selected[name].dropna()).any():
                raise ValueError(f"Non-finite {kind} statistic: {name}")
        joined = selected.merge(skeleton, on=identities, how="left", validate="many_to_one", indicator=True)
        if joined._merge.ne("both").any():
            raise ValueError(f"{kind} stats disagree with schedule identities")
        joined = joined.drop(columns="_merge")
        # No zero-fill: partial/live scores or unpublished results stay out of totals.
        joined = joined.loc[joined.points_for.notna() & joined.points_against.notna()].copy()
        if kind == "player":
            joined = joined.drop(columns=["points_for", "points_against"])
        output.append(joined.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True))
    team_rows, player_rows = output
    covered = team_rows.groupby("game_id").team.nunique()
    schedules["status"] = np.where(schedules.game_id.isin(covered.loc[covered.eq(2)].index), "Stats available",
                                    np.where(schedules.home_score.notna() | schedules.away_score.notna(), "Awaiting stats", "Scheduled"))
    return schedules.sort_values(["gameday", "game_id"]).reset_index(drop=True), team_rows, player_rows


def summarize_stats(rows, *, players=False):
    """Sum additive statistics; derive efficiency from aggregate denominators."""
    keys = ["season", "team"] + (["player_id"] if players else [])
    data = rows.loc[rows.player_id.notna()].copy() if players else rows.copy()
    metrics = [c for c in [*STAT_COLUMNS, "points_for", "points_against"] if c in data]
    grouped = data.groupby(keys, dropna=False, sort=True)
    totals = grouped[metrics].sum(min_count=1)
    totals["games_with_rows"] = grouped.game_id.nunique()
    totals = totals.reset_index()
    if players:
        names = [c for c in ("player_display_name", "position") if c in data]
        latest = data.sort_values(["gameday", "game_id"]).drop_duplicates(keys, keep="last")
        totals = totals.merge(latest[[*keys, *names]], on=keys, how="left", validate="one_to_one")
    for name, numerator, denominator in (("completion_pct", "completions", "attempts"),
                                         ("yards_per_pass_attempt", "passing_yards", "attempts"),
                                         ("yards_per_carry", "rushing_yards", "carries"),
                                         ("yards_per_reception", "receiving_yards", "receptions")):
        if numerator in totals and denominator in totals:
            totals[name] = totals[numerator] / totals[denominator].replace(0, np.nan)
            if name.endswith("pct"):
                totals[name] *= 100
    return totals


def save_snapshot(path: Path, schedules, teams, players, metadata):
    """Publish all three tables together using a single atomic ZIP replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, frame in (("schedules", schedules), ("teams", teams), ("players", players)):
                archive.writestr(f"{name}.parquet", frame.to_parquet(index=False))
            archive.writestr("manifest.json", json.dumps(metadata, indent=2, allow_nan=False))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_snapshot(path):
    with zipfile.ZipFile(path) as archive:
        frames = [pd.read_parquet(BytesIO(archive.read(name + ".parquet"))) for name in ("schedules", "teams", "players")]
        return (*frames, json.loads(archive.read("manifest.json")))
