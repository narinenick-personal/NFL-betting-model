"""Assemble the auditable team history and strictly allowlisted game predictors."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from nfl_model.data_quality import require_unique_keys
from nfl_model.features import add_pregame_features, pregame_feature_names
from nfl_model.team_games import (
    CONTEXT_COLUMNS, IDENTIFIER_COLUMNS, MARKET_SOURCE_COLUMNS,
    OBSERVED_CONTEXT_COLUMNS, TARGET_COLUMNS, build_team_game_table, prepare_schedules,
)
from nfl_model.team_metrics import TEAM_METRICS


def feature_groups(windows: Sequence[int] = (3, 5)) -> dict[str, list[str]]:
    """Disjoint column roles, persisted alongside the parquet for downstream use."""
    history = pregame_feature_names(TEAM_METRICS, windows)
    return {
        "identifiers": list(IDENTIFIER_COLUMNS),
        "football_features": list(CONTEXT_COLUMNS) + [f"{side}_{col}" for side in ("home", "away") for col in history],
        "market_features": [f"market_{col}" for col in MARKET_SOURCE_COLUMNS],
        "observed_context": list(OBSERVED_CONTEXT_COLUMNS),
        "targets": list(TARGET_COLUMNS),
    }


def select_model_features(
    games: pd.DataFrame, *, include_market: bool = False, windows: Sequence[int] = (3, 5)
) -> pd.DataFrame:
    """Explicit model inputs; never infer predictors by numeric dtype or exclusion.

    Market columns are a separate opt-in experiment. The data has no timestamped
    odds history, so these columns cannot establish an earlier executable price.
    """
    groups = feature_groups(windows)
    columns = groups["football_features"]
    if include_market:
        columns = columns + groups["market_features"]
    return games.loc[:, columns].copy()


@dataclass
class ModelingDataset:
    team_games: pd.DataFrame
    games: pd.DataFrame
    column_groups: dict[str, list[str]]


def build_modeling_dataset(
    team_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    windows: Sequence[int] = (3, 5),
) -> ModelingDataset:
    """Create one game row from both teams' same-game pregame snapshots."""
    groups = feature_groups(windows)
    schedule = prepare_schedules(schedules)
    team_games = add_pregame_features(build_team_game_table(team_stats, schedules), TEAM_METRICS, windows)
    history = pregame_feature_names(TEAM_METRICS, windows)
    games = schedule[[*IDENTIFIER_COLUMNS, *CONTEXT_COLUMNS, *OBSERVED_CONTEXT_COLUMNS,
                      *MARKET_SOURCE_COLUMNS, *TARGET_COLUMNS]].rename(
        columns={col: f"market_{col}" for col in MARKET_SOURCE_COLUMNS}
    )
    for side in ("home", "away"):
        side_history = team_games[["game_id", "team", "is_home", *history]].copy()
        side_history = side_history.loc[side_history["is_home"].eq(int(side == "home"))].drop(columns="is_home")
        side_history = side_history.rename(columns={"team": f"{side}_team", **{col: f"{side}_{col}" for col in history}})
        games = games.merge(side_history, on=["game_id", f"{side}_team"], how="left", validate="one_to_one", indicator=True)
        if not games["_merge"].eq("both").all():
            raise ValueError(f"Missing {side} pregame features")
        games = games.drop(columns="_merge")
    require_unique_keys(games, ["game_id"], "modeling games")
    ordered = [col for columns in groups.values() for col in columns]
    if set(ordered) != set(games.columns) or len(ordered) != len(set(ordered)):
        raise ValueError("Every modeling column must have exactly one role")
    games = games[ordered].sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return ModelingDataset(team_games=team_games, games=games, column_groups=groups)
