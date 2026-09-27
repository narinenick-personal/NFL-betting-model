"""Historical summaries: every predictor uses strictly prior team games."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from nfl_model.data_quality import require_columns, require_unique_keys

LEGACY_METRICS = ("points", "total_yards", "passing_yards", "rushing_yards", "turnovers")


def _sort_history(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["team", "season", "week"], "team history")
    if df[["team", "season", "week"]].isna().any().any():
        raise ValueError("team history contains missing team/season/week")
    # Calendar dates handle postponed games; week alone is only a legacy fallback.
    order = ["team", "season", "gameday" if "gameday" in df else "week"]
    if "gameday" in df:
        df = df.copy()
        df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
        # Same-day games for a team cannot be ordered safely without completion times.
        require_unique_keys(df, ["team", "season", "gameday"], "team history")
    return df.sort_values(order, kind="stable").reset_index(drop=True)


def add_team_rolling_features(
    team_stats: pd.DataFrame, window: int = 4
) -> pd.DataFrame:
    """Legacy API, now resetting rolling history at every season boundary."""
    _validate_windows((window,))
    df = _sort_history(team_stats)
    for col in LEGACY_METRICS:
        if col in df:
            df[f"{col}_rolling_{window}"] = df.groupby(["team", "season"])[col].transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )
    return df


def add_team_season_averages(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Legacy API: expanding means within a season, excluding the current game."""
    df = _sort_history(team_stats)
    for col in LEGACY_METRICS:
        if col in df:
            df[f"{col}_season_avg"] = df.groupby(["team", "season"])[col].transform(
                lambda values: values.shift(1).expanding(min_periods=1).mean()
            )
    return df


def _validate_windows(windows: Sequence[int]) -> None:
    if not windows or len(set(windows)) != len(windows) or any(
        isinstance(w, bool) or not isinstance(w, int) or w <= 0 for w in windows
    ):
        raise ValueError("windows must be distinct positive integers")


def pregame_feature_names(metrics: Sequence[str], windows: Sequence[int]) -> list[str]:
    _validate_windows(windows)
    return (
        [f"pregame_{metric}_last_{w}" for w in windows for metric in metrics]
        + [f"pregame_{metric}_season_avg" for metric in metrics]
        + ["pregame_games_played"]
        + [f"pregame_games_last_{w}" for w in windows]
    )


def add_pregame_features(
    team_games: pd.DataFrame,
    metrics: Sequence[str],
    windows: Sequence[int] = (3, 5),
) -> pd.DataFrame:
    """Mean of game-level metrics over prior games, within team and NFL season.

    Windows count games, not calendar weeks. Postseason retains the same season's
    regular-season history. Missing observations stay missing; no imputation or
    cross-season carryover occurs. Raw metrics remain for audit only.
    """
    _validate_windows(windows)
    require_unique_keys(team_games, ["game_id", "team"], "team games")
    require_columns(team_games, list(metrics), "team games")
    df = _sort_history(team_games)
    groups = df.groupby(["team", "season"], sort=False)
    summaries: dict[str, pd.Series] = {}
    for window in windows:
        for metric in metrics:
            summaries[f"pregame_{metric}_last_{window}"] = groups[metric].transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )
    for metric in metrics:
        summaries[f"pregame_{metric}_season_avg"] = groups[metric].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
    summaries["pregame_games_played"] = groups.cumcount()
    for window in windows:
        summaries[f"pregame_games_last_{window}"] = summaries["pregame_games_played"].clip(upper=window)
    return pd.concat([df, pd.DataFrame(summaries, index=df.index)], axis=1)
