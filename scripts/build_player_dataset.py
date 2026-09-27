"""Build player forecasts from lagged rosters and reconcile with team outcomes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from nfl_model.data_loader import load_player_stats, load_weekly_rosters, load_snap_counts, load_players, load_team_stats
from nfl_model.market_reporting import file_hash
from nfl_model.player_dataset import build_player_dataset


def main():
    seasons = list(range(2021, 2026))
    result = build_player_dataset(load_player_stats(seasons), load_weekly_rosters(seasons), load_snap_counts(seasons),
                                  load_players(), pd.read_parquet(ROOT / "data/processed/team_game_dataset.parquet"), load_team_stats(seasons))
    output = ROOT / "data/processed"
    for name, frame in (("player_modeling_dataset", result.players), ("player_game_observations", result.observations),
                        ("player_reconciliation", result.reconciliation), ("player_id_exceptions", result.id_exceptions)):
        frame.to_parquet(output / f"{name}.parquet", index=False)
    result.metadata["input_sha256"] = {p.name: file_hash(p) for p in (ROOT / "data/raw").glob("*.parquet") if p.name.endswith("2021_2022_2023_2024_2025.parquet") or p.name == "players.parquet"}
    result.metadata["dataset_sha256"] = file_hash(output / "player_modeling_dataset.parquet")
    (output / "player_modeling_manifest.json").write_text(json.dumps(result.metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.metadata.items() if key not in ("feature_columns", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
