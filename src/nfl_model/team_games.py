"""Validated schedule/stat joins, with one record per team per completed game."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data_quality import require_columns, require_unique_keys
from nfl_model.team_metrics import RAW_METRICS, RATE_METRICS, SUM_METRICS, add_team_game_metrics

GAME_TYPES = ("REG", "WC", "DIV", "CON", "SB")
MARKET_SOURCE_COLUMNS = (
    "away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
    "home_spread_odds", "total_line", "under_odds", "over_odds",
)
CONTEXT_COLUMNS = (
    "home_rest", "away_rest", "neutral_site", "div_game", "surface",
    "home_is_home", "away_is_home", "week", "game_type",
)
# Recorded weather and actual roof status do not establish what was known at bet time.
OBSERVED_CONTEXT_COLUMNS = ("roof", "temp", "wind")
IDENTIFIER_COLUMNS = ("game_id", "season", "gameday", "home_team", "away_team")
TARGET_COLUMNS = ("home_score", "away_score", "game_total", "home_margin")


def prepare_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    """Keep completed regular/postseason games and normalize optional context."""
    required = [
        "game_id", "season", "week", "game_type", "gameday",
        "home_team", "away_team", "home_score", "away_score",
    ]
    require_columns(schedules, required, "schedules")
    require_unique_keys(schedules, ["game_id"], "schedules")
    df = schedules.loc[schedules["game_type"].isin(GAME_TYPES)].copy()
    for score in ("home_score", "away_score"):
        df[score] = pd.to_numeric(df[score], errors="raise")
        valid = df[score].dropna()
        if ((valid < 0) | (valid % 1 != 0) | ~np.isfinite(valid)).any():
            raise ValueError(f"schedules has invalid {score}")
    if (df["home_score"].isna() ^ df["away_score"].isna()).any():
        raise ValueError("schedules has partially missing game scores")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    if df.empty:
        raise ValueError("No completed regular/postseason games in schedules")
    if df[required].isna().any().any():
        raise ValueError("Completed schedules contain missing required fields")
    if (df["home_team"] == df["away_team"]).any():
        raise ValueError("Home team and away team must differ")
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    if df["gameday"].isna().any():
        raise ValueError("Completed schedules contain missing game dates")
    for col in ("season", "week"):
        values = pd.to_numeric(df[col], errors="raise")
        if ((values <= 0) | (values % 1 != 0)).any():
            raise ValueError(f"Invalid schedule {col}")
        df[col] = values.astype(int)
    for col in ("location", "surface", "roof"):
        if col not in df:
            df[col] = pd.NA
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    unknown_locations = df["location"].dropna().loc[lambda s: ~s.isin(["Home", "Neutral"])]
    if not unknown_locations.empty:
        raise ValueError(f"Unknown schedule locations: {unknown_locations.unique().tolist()}")
    df["neutral_site"] = df["location"].eq("Neutral").astype("Int64")
    df["home_is_home"] = 1  # Designation; neutral_site encodes actual venue advantage.
    df["away_is_home"] = 0
    for col in ("home_rest", "away_rest", "div_game", "temp", "wind", *MARKET_SOURCE_COLUMNS):
        df[col] = pd.to_numeric(df[col], errors="raise") if col in df else np.nan
        if np.isinf(df[col].dropna()).any():
            raise ValueError(f"Non-finite schedule {col}")
    df["season_type"] = np.where(df["game_type"].eq("REG"), "REG", "POST")
    df["game_total"] = df["home_score"] + df["away_score"]
    df["home_margin"] = df["home_score"] - df["away_score"]
    # Recompute targets from scores; do not depend on result/total aliases.
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _join_team_stats(stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Internal join against an already validated, completed-game schedule."""
    require_columns(
        stats, ["game_id", "team", "season", "week", "season_type", "opponent_team"], "team stats"
    )
    require_unique_keys(stats, ["game_id", "team"], "team stats")
    pieces = []
    shared = ["game_id", "season", "week", "game_type", "season_type", "gameday", "neutral_site",
              "div_game", "roof", "surface", "temp", "wind"]
    for side, other in (("home", "away"), ("away", "home")):
        part = schedules[shared].copy()
        part["team"] = schedules[f"{side}_team"]
        part["opponent_team"] = schedules[f"{other}_team"]
        part["is_home"] = int(side == "home")
        part["rest"] = schedules[f"{side}_rest"]
        part["opponent_rest"] = schedules[f"{other}_rest"]
        part["points_for"] = schedules[f"{side}_score"]
        part["points_against"] = schedules[f"{other}_score"]
        pieces.append(part)
    skeleton = pd.concat(pieces, ignore_index=True)
    require_unique_keys(skeleton, ["game_id", "team"], "schedule team games")
    identities = ["season", "week", "season_type", "opponent_team"]
    # An allowlist prevents accidental use of arbitrary future columns added upstream.
    sources = set(RAW_METRICS.values()) - {"points_for", "points_against"}
    sources.update(source for columns in SUM_METRICS.values() for source in columns)
    sources.update(source for pair in RATE_METRICS.values() for source in pair)
    columns = ["game_id", "team", *identities, *sorted(sources.intersection(stats.columns) - set(identities))]
    selected = stats[columns].rename(columns={col: f"stats_{col}" for col in identities})
    joined = skeleton.merge(selected, on=["game_id", "team"], how="outer", validate="one_to_one", indicator=True)
    unmatched = joined["_merge"].ne("both")
    if unmatched.any():
        examples = joined.loc[unmatched, ["game_id", "team", "_merge"]].head(8).to_dict("records")
        raise ValueError(f"Unmatched completed schedule/team stats records: {examples}")
    for col in identities:
        bad = joined[f"stats_{col}"].isna() | joined[col].ne(joined[f"stats_{col}"])
        if bad.any():
            raise ValueError(f"Team stats {col} disagrees with schedule: {joined.loc[bad, 'game_id'].tolist()[:5]}")
    joined = joined.drop(columns=["_merge", *(f"stats_{col}" for col in identities)])
    require_unique_keys(joined, ["team", "season", "gameday"], "team games")
    return add_team_game_metrics(joined).sort_values(["gameday", "game_id", "is_home"]).reset_index(drop=True)


def build_team_game_table(team_stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Join completed games strictly; ignore stats only for known excluded games.

    Both inputs must cover the same seasons. Unknown game IDs are errors, while
    scheduled unplayed/cancelled/preseason games never enter historical features.
    """
    completed = prepare_schedules(schedules)
    require_columns(team_stats, ["game_id", "team"], "team stats")
    require_unique_keys(team_stats, ["game_id", "team"], "team stats")
    unknown = ~team_stats["game_id"].isin(schedules["game_id"])
    if unknown.any():
        raise ValueError(f"Team stats reference unknown schedule games: {team_stats.loc[unknown, 'game_id'].head().tolist()}")
    selected = team_stats.loc[team_stats["game_id"].isin(completed["game_id"])].copy()
    return _join_team_stats(selected, completed)
