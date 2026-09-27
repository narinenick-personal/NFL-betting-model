"""Replay an actual supplied archive of 2025 game-day player-prop quotes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from nfl_model.market_reporting import file_hash
from nfl_model.quote_backtesting import validate_quotes, backtest_quotes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quotes", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/quote_replay")
    args = parser.parse_args()
    model = joblib.load(ROOT / "artifacts/players/models.joblib")["football_players"]
    games = pd.read_parquet(ROOT / "data/processed/game_modeling_dataset.parquet")
    players = pd.read_parquet(ROOT / "data/processed/player_modeling_dataset.parquet")
    schedules = pd.read_parquet(ROOT / "data/raw/schedules_2021_2022_2023_2024_2025.parquet")
    quotes = validate_quotes(pd.read_csv(args.quotes), schedules, players, fit_end=model.fit_end)
    def provider(game_id):
        return next(model.simulate(games.loc[games.game_id.eq(game_id)], players.loc[players.game_id.eq(game_id)]))[3]
    results, summary = backtest_quotes(quotes, players, provider)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output_dir / "results.parquet", index=False)
    results.to_csv(args.output_dir / "results.csv", index=False)
    summary.update({"quote_file_sha256": file_hash(args.quotes), "model_sha256": file_hash(ROOT / "artifacts/players/models.joblib"),
                    "protocol": "Every supplied quote, supplied stakes, no outcome-based selection. ROI includes stakes returned on pushes/voids. Source features are retrospective snapshots.",
                    "clv": "Positive line CLV means a favorable entry threshold. Price CLV is closing minus entry raw implied probability, in percentage points, only at the same line. Supplied closing snapshots are not independently verified as market close."})
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
