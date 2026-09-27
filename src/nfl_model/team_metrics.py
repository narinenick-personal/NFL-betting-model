"""Explicit game-level definitions. These observations must be lagged before use."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Output name -> nflverse source. Missing source columns produce NaN, never zero.
RAW_METRICS = {
    "points_for": "points_for",
    "points_against": "points_against",
    "pass_attempts": "attempts",
    "rush_attempts": "carries",
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "passing_epa": "passing_epa",
    "rushing_epa": "rushing_epa",
    "cpoe": "passing_cpoe",
    "sacks_allowed": "sacks_suffered",
    "interceptions_thrown": "passing_interceptions",
    "team_fumbles_lost": "fumbles_lost_total",
    "penalties_per_game": "penalties",
    "penalty_yards_per_game": "penalty_yards",
    "defensive_sacks": "def_sacks",
    "defensive_interceptions": "def_interceptions",
    "defensive_qb_hits": "def_qb_hits",
}

SUM_METRICS = {
    "pass_dropbacks": ("attempts", "sacks_suffered"),
    "total_offensive_plays": ("attempts", "carries", "sacks_suffered"),
    "offensive_fumbles_lost": ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
    "turnovers": ("passing_interceptions", "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
    "offensive_first_downs": ("passing_first_downs", "rushing_first_downs"),
    "offensive_touchdowns": ("passing_tds", "rushing_tds"),
}

# Numerators can be source columns or earlier derived metrics.
RATE_METRICS = {
    "passing_yards_per_attempt": ("passing_yards", "pass_attempts"),
    "rushing_yards_per_carry": ("rushing_yards", "rush_attempts"),
    "passing_epa_per_attempt": ("passing_epa", "pass_attempts"),
    "passing_epa_per_dropback": ("passing_epa", "pass_dropbacks"),
    "rushing_epa_per_carry": ("rushing_epa", "rush_attempts"),
    "sack_rate": ("sacks_allowed", "pass_dropbacks"),
    "interception_rate": ("interceptions_thrown", "pass_attempts"),
    "fumble_lost_rate": ("offensive_fumbles_lost", "total_offensive_plays"),
    "turnover_rate": ("turnovers", "total_offensive_plays"),
    "offensive_first_down_rate": ("offensive_first_downs", "total_offensive_plays"),
    "explosive_passing_rate": ("passing_20", "pass_attempts"),
    "explosive_rushing_rate": ("rushing_10", "rush_attempts"),
    "offensive_scoring_efficiency": ("offensive_touchdowns", "total_offensive_plays"),
}

TEAM_METRICS = tuple(RAW_METRICS) + tuple(SUM_METRICS) + tuple(RATE_METRICS)


def metric_definitions() -> dict[str, str]:
    return {
        **{name: f"nflverse {source}" for name, source in RAW_METRICS.items()},
        **{name: " + ".join(sources) for name, sources in SUM_METRICS.items()},
        **{name: f"{num} / {den} (NaN if denominator <= 0)" for name, (num, den) in RATE_METRICS.items()},
        "points_for": "Schedule score for this team, including all scoring phases",
        "points_against": "Schedule score for the opponent, including all scoring phases",
    }


def add_team_game_metrics(team_games: pd.DataFrame) -> pd.DataFrame:
    """Compute raw observations without silently fabricating unavailable metrics."""
    df = team_games.copy()

    def numeric(name: str) -> pd.Series:
        if name not in df:
            return pd.Series(np.nan, index=df.index, dtype=float)
        values = pd.to_numeric(df[name], errors="raise").astype(float)
        if np.isinf(values).any():
            raise ValueError(f"Non-finite values in {name}")
        return values

    for name, source in RAW_METRICS.items():
        df[name] = numeric(source)
    for name, sources in SUM_METRICS.items():
        # Addition propagates missing components; sum(skipna=True) would not.
        values = numeric(sources[0])
        for source in sources[1:]:
            values = values + numeric(source)
        df[name] = values
    for name, (numerator, denominator) in RATE_METRICS.items():
        den = numeric(denominator)
        df[name] = numeric(numerator).div(den.where(den > 0))
    return df
