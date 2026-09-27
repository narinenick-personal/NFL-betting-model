"""Train chronological team volume and player opportunity benchmarks."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from nfl_model.market_reporting import file_hash
from nfl_model.opportunity_models import build_volume_games
from nfl_model.opportunity_backtesting import run_opportunity_backtest


def main():
    paths = {name: ROOT / "data/processed" / filename for name, filename in (
        ("players", "player_modeling_dataset.parquet"), ("games", "game_modeling_dataset.parquet"), ("teams", "team_game_dataset.parquet"))}
    manifest = json.loads((ROOT / "data/processed/player_modeling_manifest.json").read_text())
    if file_hash(paths["players"]) != manifest["dataset_sha256"]:
        raise ValueError("Player dataset hash disagrees with its manifest")
    players = pd.read_parquet(paths["players"])
    games = build_volume_games(pd.read_parquet(paths["games"]), pd.read_parquet(paths["teams"]))
    run = run_opportunity_backtest(games, players, progress=lambda message: print(message, flush=True))
    output = ROOT / "artifacts/opportunities"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("volume_predictions", "player_predictions", "volume_metrics", "player_metrics", "participation_metrics", "tuning"):
        getattr(run, name).to_parquet(output / f"{name}.parquet", index=False)
    joblib.dump(run.fold_models, output / "fold_models.joblib", compress=3)
    restored = joblib.load(output / "fold_models.joblib")
    for season, bundle in restored.items():
        forecast = games.loc[games.season.eq(season)]
        expected = run.volume_predictions.loc[lambda d: d.season.eq(season) & d.model.eq("ridge")].pivot(index="game_id", columns="target", values="predicted")
        got = bundle["volume"].predict(forecast)
        if not np.allclose(got, expected.loc[forecast.game_id, got.columns], atol=1e-10):
            raise ValueError("Volume bundle failed reload check")
        forecast_players = players.loc[players.season.eq(season)]
        shares = bundle["players"].predict_shares(forecast_players)
        expected = run.player_predictions.loc[lambda d: d.season.eq(season) & d.model.eq("trained")].pivot(index=["game_id", "team", "player_id"], columns="target", values="predicted_share")
        aligned = expected.reindex(pd.MultiIndex.from_frame(forecast_players[["game_id", "team", "player_id"]]))
        if not np.allclose(shares, aligned[shares.columns], atol=1e-10):
            raise ValueError("Player share bundle failed reload check")
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "audit_season": 2025,
                "protocol": "2022–2025 rolling-origin fits, earlier seasons only. Team-volume alpha tuned on latest earlier season (initial fold date-split). Player alpha=.05 and logistic C=1 fixed. Audit was previously inspected.",
                "inputs": {name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()},
                "selected_alphas": {str(s): b["selected_alpha"] for s, b in restored.items()},
                "model_sha256": file_hash(output / "fold_models.joblib"),
                "source_sha256": {p.name: file_hash(p) for p in (ROOT / "src/nfl_model").glob("*.py")},
                "reload_check": "Every fold's volume predictions and player shares reproduced",
                "limitations": [*manifest["limitations"], "Point share forecasts normalize expected active weights; the mean of stochastic normalized allocations can differ.", "Participation is modeled without current injury/active-list information."]}
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    lines = ["# Team volume and player opportunity validation", "", metadata["protocol"], "",
             "The trained model predicts offensive participation and conditional player shares. All player shares sum to one per team and opportunity, including OTHER. Predicted counts use forecast team volume, never observed test-game totals.", "",
             "## 2025 team volume", "", "| Model | Target | MAE | RMSE |", "| --- | --- | ---: | ---: |"]
    for row in run.volume_metrics.loc[run.volume_metrics.season.eq(2025)].itertuples():
        lines.append(f"| {row.model} | {row.target} | {row.mae:.3f} | {row.rmse:.3f} |")
    lines.extend(["", "## 2025 players with prior usage", "", "The cohort requires positive last-five usage before the game, not current participation.", "",
                  "| Model | Target | Rows | MAE | RMSE | Bias |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    selected = run.player_metrics.loc[run.player_metrics.season.eq(2025) & run.player_metrics.cohort.eq("prior_usage")]
    for row in selected.itertuples():
        lines.append(f"| {row.model} | {row.target} | {row.rows} | {row.mae:.3f} | {row.rmse:.3f} | {row.bias:.3f} |")
    lines.extend(["", "## 2025 offensive participation", "", "| Model | Rows | Brier | Log loss | Predicted active | Observed active |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in run.participation_metrics.loc[run.participation_metrics.season.eq(2025)].itertuples():
        lines.append(f"| {row.model} | {row.rows} | {row.brier:.4f} | {row.log_loss:.4f} | {row.predicted_active:.1%} | {row.actual_active:.1%} |")
    lines.extend(["", "## Limits", "", *[f"- {item}" for item in metadata["limitations"]], "", metadata["reload_check"], ""])
    (output / "report.md").write_text("\n".join(lines))
    print(selected.round(4).to_string(index=False))
    print(run.participation_metrics.loc[run.participation_metrics.season.eq(2025)].round(4).to_string(index=False))
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
