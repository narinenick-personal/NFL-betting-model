"""Refresh observed season stats without changing model datasets or fitted models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

# This is a separate CLI process: explicit refreshes bypass nflreadpy's own cache.
os.environ["NFLREADPY_CACHE"] = "off"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from nfl_model import data_loader
from nfl_model.season_stats import prepare_season_stats, read_snapshot, save_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2021, 2027)))
    parser.add_argument("--refresh", action="store_true", help="Download each selected season again, bypassing both caches")
    parser.add_argument("--output", type=Path, default=ROOT / "data/season_stats/stats.zip")
    args = parser.parse_args()
    seasons = sorted(set(args.seasons))
    if not seasons or min(seasons) < 2021 or max(seasons) > datetime.now().year:
        parser.error("Choose NFL seasons from 2021 through the current year")
    frames, sources = {}, {}
    for name in ("schedules", "team_stats", "player_stats"):
        pieces = []
        history_path = data_loader.DEFAULT_CACHE_DIR / f"{name}_2021_2022_2023_2024_2025.parquet"
        history = None
        for season in seasons:
            print(f"Loading {name} {season}...", flush=True)
            single = data_loader.DEFAULT_CACHE_DIR / f"{name}_{season}.parquet"
            if not args.refresh and not single.exists() and season <= 2025 and history_path.exists():
                if history is None:
                    history = pd.read_parquet(history_path)
                data, source = history.loc[history.season.eq(season)].copy(), history_path
            else:
                data = getattr(data_loader, "load_" + name)([season], refresh=args.refresh)
                source = single
            if name == "schedules" and data.empty:
                raise ValueError(f"No {season} schedule is available; keeping the existing snapshot")
            if not data.empty and not data.season.eq(season).all():
                raise ValueError(f"Unexpected season in {name}")
            sources[f"{name}_{season}"] = {"cache_file": source.name, "rows": len(data),
                "cache_written_at_utc": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            pieces.append(data)
        frames[name] = pd.concat(pieces, ignore_index=True)
    schedule, teams, players = prepare_season_stats(frames["schedules"], frames["team_stats"], frames["player_stats"])
    if args.output.exists():
        previous = read_snapshot(args.output)
        schedule, teams, players = [pd.concat([old.loc[~old.season.isin(seasons)], new], ignore_index=True)
                                   for old, new in zip(previous[:3], (schedule, teams, players))]
        sources = {**previous[3].get("sources", {}), **sources}
    coverage = {str(season): {"scheduled_games": len(rows), "games_with_team_stats": int(rows.status.eq("Stats available").sum()),
                              "team_rows": int(teams.season.eq(season).sum()), "player_rows": int(players.season.eq(season).sum()),
                              "weeks_with_stats": sorted(int(w) for w in teams.loc[teams.season.eq(season), "week"].unique())}
                for season, rows in schedule.groupby("season")}
    metadata = {"built_at_utc": datetime.now(timezone.utc).isoformat(), "seasons": sorted(int(s) for s in schedule.season.unique()),
                "coverage": coverage, "sources": sources,
                "scope": "Observed statistics only. Not loaded by the historical forecast models. NFL season labels, not calendar-year filtering.",
                "source_url": "https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html"}
    save_snapshot(args.output, schedule, teams, players, metadata)
    print(json.dumps(coverage, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
