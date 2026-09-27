"""Build reports and a durable feature/target contract for saved datasets."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from nfl_model.modeling_dataset import ModelingDataset
from nfl_model.team_metrics import TEAM_METRICS, metric_definitions


def dataset_report(dataset: ModelingDataset, *, tests: str = "not run by this build") -> dict:
    games = dataset.games
    groups = dataset.column_groups
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "games": len(games),
        "team_games": len(dataset.team_games),
        "columns": len(games.columns),
        "football_features": len(groups["football_features"]),
        "market_features": len(groups["market_features"]),
        "observed_context_columns": len(groups["observed_context"]),
        "seasons": sorted(int(s) for s in games["season"].unique()),
        "games_by_season": {str(k): int(v) for k, v in games.groupby("season").size().items()},
        "games_by_type": {str(k): int(v) for k, v in games.groupby("game_type").size().items()},
        "missing_pct_by_group": {
            group: float(games[cols].isna().to_numpy().mean() * 100)
            for group, cols in groups.items()
        },
        "missing_pct_by_column": (games.isna().mean() * 100).to_dict(),
        "raw_metric_missing_pct": (dataset.team_games[list(TEAM_METRICS)].isna().mean() * 100).to_dict(),
        "all_missing_football_features": [col for col in groups["football_features"] if games[col].isna().all()],
        "duplicate_games": int(games.duplicated("game_id").sum()),
        "duplicate_team_games": int(dataset.team_games.duplicated(["game_id", "team"]).sum()),
        "tests": tests,
    }


def dataset_manifest(dataset: ModelingDataset, windows: tuple[int, ...]) -> dict:
    return {
        "schema_version": 1,
        "column_groups": dataset.column_groups,
        "team_metrics": metric_definitions(),
        "rolling_windows": list(windows),
        "history_policy": "Within team and NFL season; chronological gameday; shift(1); min_periods=1; no imputation; regular season carries into postseason only.",
        "rate_aggregation": "Arithmetic mean of prior game rates, not pooled numerator/denominator rates. Missing observations are skipped within the fixed game window.",
        "scoring_efficiency": "Offensive passing plus rushing touchdowns per offensive play; not points per drive or red-zone efficiency.",
        "opportunity_definition": "Pass attempts + carries + sacks suffered; excludes penalty-only/no-play snaps and two-point tries.",
        "fumble_definition": "Sack, rushing and receiving fumbles lost count as offensive losses; all-phase team_fumbles_lost is retained separately.",
        "explosive_definition": "passing_20 / attempts; rushing_10 / carries (20+ passing yards, 10+ rushing yards).",
        "passing_epa_caveat": "Source passing EPA includes sacks; per_attempt divides that supplied total by attempts. The additional per_dropback metric divides by attempts + sacks.",
        "market_policy": "market_ prefix; excluded by default. Positive market_spread_line means home favored; no timestamped price history, so no executable early-line or CLV claims.",
        "observed_context_policy": "Recorded temp, wind and actual roof status retained, excluded by default; replace with timestamped pregame observations/forecasts before using in an earlier-decision backtest.",
        "source_vintage_policy": "Current nflverse snapshots, not point-in-time archives. Upstream corrections and retrospective EPA/CPOE models can affect historical values; strict as-of backtests need archived vintages.",
        "sources": {
            "team_stats": "https://nflreadr.nflverse.com/articles/dictionary_team_stats.html",
            "schedules": "https://nflreadr.nflverse.com/articles/dictionary_schedules.html",
        },
    }


def format_report(dataset: ModelingDataset, report: dict) -> str:
    groups = dataset.column_groups
    missing = pd.Series(report["missing_pct_by_column"]).sort_values(ascending=False)
    preview = ["game_id", "home_team", "away_team", "home_score", "away_score", "game_total", "home_margin",
               "home_pregame_points_for_last_3", "away_pregame_points_for_last_3"]
    lines = [
        f"Games: {report['games']:,} | Team-game rows: {report['team_games']:,}",
        f"Features: {report['football_features']} football + {report['market_features']} optional market",
        f"Other columns: {len(groups['identifiers'])} identifiers, {len(groups['targets'])} targets, {len(groups['observed_context'])} observed context",
        f"Seasons: {', '.join(map(str, report['seasons']))}",
        f"Games by season: {report['games_by_season']}",
        f"Duplicate game / team-game keys: {report['duplicate_games']} / {report['duplicate_team_games']}",
        "Missing cells by column group:",
        *(f"  {group}: {pct:.2f}%" for group, pct in report["missing_pct_by_group"].items()),
        "Highest per-column missing percentages (full details in JSON report):",
        missing.head(8).round(2).to_string(),
        "First five games (selected columns):",
        dataset.games[preview].head().to_string(index=False),
        f"Tests: {report['tests']}",
    ]
    if report["all_missing_football_features"]:
        lines.append(f"Unavailable features: {report['all_missing_football_features']}")
    return "\n".join(lines)
