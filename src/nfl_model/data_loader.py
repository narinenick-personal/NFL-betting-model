from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nflreadpy as nfl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw"


def _as_pandas(df) -> pd.DataFrame:
    """Convert an nflreadpy Polars DataFrame to pandas."""
    return df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)


def _cache_path(name: str, seasons: Iterable[int] | None = None) -> Path:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if seasons is not None:
        suffix = "_" + "_".join(str(s) for s in seasons)
    return DEFAULT_CACHE_DIR / f"{name}{suffix}.parquet"


def _normalize_seasons(seasons: int | list[int]) -> list[int]:
    values = [seasons] if isinstance(seasons, int) else list(seasons)
    if not values or any(isinstance(s, bool) or not isinstance(s, int) for s in values):
        raise ValueError("seasons must contain one or more integer years")
    return sorted(set(values))


def load_schedules(seasons: int | list[int], *, refresh: bool = False) -> pd.DataFrame:
    seasons_list = _normalize_seasons(seasons)
    path = _cache_path("schedules", seasons_list)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = _as_pandas(nfl.load_schedules(seasons_list))
    df.to_parquet(path, index=False)
    return df


def load_player_stats(seasons: int | list[int], *, refresh: bool = False) -> pd.DataFrame:
    seasons_list = _normalize_seasons(seasons)
    path = _cache_path("player_stats", seasons_list)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = _as_pandas(nfl.load_player_stats(seasons_list))
    df.to_parquet(path, index=False)
    return df


def load_team_stats(seasons: int | list[int], *, refresh: bool = False) -> pd.DataFrame:
    seasons_list = _normalize_seasons(seasons)
    path = _cache_path("team_stats", seasons_list)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = _as_pandas(nfl.load_team_stats(seasons_list))
    df.to_parquet(path, index=False)
    return df


def load_players() -> pd.DataFrame:
    path = _cache_path("players")
    if path.exists():
        return pd.read_parquet(path)

    df = _as_pandas(nfl.load_players())
    df.to_parquet(path, index=False)
    return df


def load_weekly_rosters(seasons: int | list[int], *, refresh: bool = False) -> pd.DataFrame:
    seasons_list = _normalize_seasons(seasons)
    path = _cache_path("weekly_rosters", seasons_list)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = _as_pandas(nfl.load_rosters_weekly(seasons_list))
    df.to_parquet(path, index=False)
    return df


def load_snap_counts(seasons: int | list[int], *, refresh: bool = False) -> pd.DataFrame:
    seasons_list = _normalize_seasons(seasons)
    path = _cache_path("snap_counts", seasons_list)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = _as_pandas(nfl.load_snap_counts(seasons_list))
    df.to_parquet(path, index=False)
    return df


def load_injuries(seasons: int | list[int]) -> pd.DataFrame:
    seasons_list = [seasons] if isinstance(seasons, int) else list(seasons)
    path = _cache_path("injuries", seasons_list)
    if path.exists():
        return pd.read_parquet(path)

    df = _as_pandas(nfl.load_injuries(seasons_list))
    df.to_parquet(path, index=False)
    return df
