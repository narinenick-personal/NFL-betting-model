"""Validate correlated player simulations and fixed research thresholds."""
from __future__ import annotations

import argparse
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
from nfl_model.player_backtesting import run_player_simulation_backtest
from nfl_model.player_dataset import PLAYER_TARGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("At least two draws are required")
    opportunity_dir = ROOT / "artifacts/opportunities"
    opportunities = json.loads((opportunity_dir / "run.json").read_text())
    for identity in opportunities["inputs"].values():
        if file_hash(Path(identity["path"])) != identity["sha256"]:
            raise ValueError("Opportunity input changed; regenerate its fits")
    if file_hash(opportunity_dir / "fold_models.joblib") != opportunities["model_sha256"]:
        raise ValueError("Opportunity model bundle hash mismatch")
    market_dir = ROOT / "artifacts/markets"
    market_meta = json.loads((market_dir / "run.json").read_text())
    if file_hash(market_dir / "models.joblib") != market_meta["artifact_sha256"]["models.joblib"]:
        raise ValueError("Score model bundle hash mismatch")
    games = pd.read_parquet(ROOT / "data/processed/game_modeling_dataset.parquet")
    players = pd.read_parquet(ROOT / "data/processed/player_modeling_dataset.parquet")
    teams = pd.read_parquet(ROOT / "data/processed/team_game_dataset.parquet")
    errors = pd.read_parquet(ROOT / "artifacts/distributions/residuals.parquet")
    volume = pd.read_parquet(opportunity_dir / "volume_predictions.parquet")
    run = run_player_simulation_backtest(games, players, teams, errors.loc[errors.model.eq("ridge_football")],
                                         volume.loc[volume.model.eq("ridge")], joblib.load(opportunity_dir / "fold_models.joblib"),
                                         final_score_model=joblib.load(market_dir / "models.joblib")["ridge_football"],
                                         draws=args.draws, seed=args.seed, progress=lambda message: print(message, flush=True))
    output = ROOT / "artifacts/players"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("diagnostics", "metrics", "production_predictions", "fit_log", "example_draws"):
        getattr(run, name).to_parquet(output / f"{name}.parquet", index=False)
    pits = run.diagnostics.assign(bin=np.minimum((run.diagnostics.pit * 10).astype(int), 9)).groupby(["season", "statistic", "bin"]).size().rename("games").reset_index()
    pits.to_parquet(output / "pit_histograms.parquet", index=False)
    joblib.dump(run.models, output / "models.joblib", compress=3)
    model = joblib.load(output / "models.joblib")["football_players"]
    example_id = run.example_draws.game_id.iloc[0]
    example_games = games.loc[games.game_id.eq(example_id)].drop(columns=["home_score", "away_score", "game_total", "home_margin"])
    example_players = players.loc[players.game_id.eq(example_id)].drop(columns=[*PLAYER_TARGETS, "offense_active", "offense_snaps", "offense_pct", *(f"team_{col}" for col in PLAYER_TARGETS)])
    for _, _, _, records in model.simulate(example_games, example_players, draws=args.draws, seed=args.seed):
        for team, record in records.items():
            for i, row in enumerate(record["candidates"].itertuples()):
                expected = run.example_draws.loc[lambda d: d.team.eq(team) & d.player_id.eq(row.player_id)].sort_values("draw_id")
                for stat, array in record["samples"].items():
                    if not np.array_equal(array[:, i], expected[stat]):
                        raise ValueError("Reloaded player simulation is not reproducible")
    source_paths = [ROOT / "data/processed/player_modeling_dataset.parquet", ROOT / "data/processed/game_modeling_dataset.parquet",
                    ROOT / "data/processed/team_game_dataset.parquet", ROOT / "artifacts/distributions/residuals.parquet",
                    opportunity_dir / "fold_models.joblib", opportunity_dir / "volume_predictions.parquet", market_dir / "models.joblib"]
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "draws": args.draws, "seed": args.seed,
                "audit_season": 2025, "evaluation_games": int(games.loc[games.season.between(2023, 2025)].game_id.nunique()),
                "protocol": "2022 warmup out-of-time predictions. 2023–2025 expanding chronological evaluation; uncertainty calibrated only on earlier OOT errors. 2025 is a previously inspected retrospective audit.",
                "thresholds": "Fixed research thresholds; no historical player-prop quotes are represented. Candidate/diagnostic cohorts use prior usage only.",
                "reconciliation": "Every draw checked: attempts/carries sum to team volume, targets <= attempts, completions = receptions, passing yards = receiving yards, passing TDs = receiving TDs, capacity bounds, zero inactive production, offensive TDs <= score/6.",
                "reload_check": "All saved example player arrays reproduced with all actual outcomes removed from model inputs",
                "inputs": {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths},
                "artifacts": {p.name: file_hash(p) for p in output.iterdir() if p.suffix in (".parquet", ".joblib")},
                "source_sha256": {p.name: file_hash(p) for p in (ROOT / "src/nfl_model").glob("*.py")},
                "limitations": [
                    "Historical snapshots and lagged rosters cannot substitute for archived pregame injury/active-list information. Season openers have OTHER-only projections.",
                    "Independent player participation hurdles and conditional Dirichlet allocations are approximations; QB substitutions and injuries within games are not modeled explicitly.",
                    "Score/volume dependence is estimated from earlier joint residuals. Counts use censored Gaussian residuals with stochastic rounding; no possession or field-position simulation is claimed.",
                    "Yardage uses aggregate-game moment estimates with shared team errors; per-play tails and passing lateral credits are approximated. Signed integer yards are bounded by 99 times opportunities.",
                    "Touchdowns use a historical score-conditioned kernel and pooled passing fraction. Player red-zone roles and drive-level scoring are not separately fitted.",
                    "Research-threshold outcomes count nonparticipation as zero. Offensive-snap-required void contracts are available explicitly, but do not reconstruct sportsbook any-snap rules.",
                    "Monte Carlo probability estimates have sampling error. No historical prop ROI or closing-line value can be claimed without timestamped quotes and actual settlement contracts.",
                    "No tuning or model selection uses the 2025 audit. Calibration and forecast skill must be assessed from the saved metrics; simulation consistency is not proof of accuracy.",
                ]}
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    audit = run.metrics.loc[run.metrics.season.eq(2025)]
    lines = ["# Correlated player simulation audit", "", metadata["protocol"], "", f"{args.draws:,} draws per game across {metadata['evaluation_games']} games.", "",
             metadata["reconciliation"], "", "## 2025 production", "",
             "Cohorts require prior usage, not current-game participation. Missing/new players remain in OTHER. Lower MAE, RMSE and CRPS are better.", "",
             "| Statistic | Simulation RMSE | Recent-average RMSE | CRPS | Interval coverage | Model interval mass | 80% width | Rows |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for stat, group in audit.groupby("statistic", sort=False):
        sim = group.loc[group.model.eq("simulation")].iloc[0]
        recent = group.loc[group.model.eq("recent_average")].iloc[0]
        lines.append(f"| {stat} | {sim.rmse:.3f} | {recent.rmse:.3f} | {sim.crps:.3f} | {sim.coverage_80:.1%} | {sim.interval_mass_80:.1%} | {sim.width_80:.2f} | {sim.rows} |")
    lines.extend(["", "Intervals use the 10th and 90th percentiles. Discrete scores and the nonparticipation atom can make their implied probability exceed 80%; compare empirical coverage with model interval mass. Randomized PIT histograms provide a second view of calibration, with randomization inside atoms. Neither comparison alone proves calibration."])
    lines.extend(["", "## Fixed research thresholds", "", metadata["thresholds"], "",
                  "The position-frequency comparator estimates the earlier over-rate for that position and prior-usage cohort, with a half-count prior. It is a simple probability baseline, not a sportsbook price.", "",
                  "| Statistic | Simulation Brier | Position Brier | Simulation log loss | Position log loss |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for stat, group in audit.groupby("statistic", sort=False):
        sim = group.loc[group.model.eq("simulation")].iloc[0]
        baseline = group.loc[group.model.eq("position_frequency")].iloc[0]
        lines.append(f"| {stat} | {sim.brier:.4f} | {baseline.brier:.4f} | {sim.log_loss:.4f} | {baseline.log_loss:.4f} |")
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in metadata["limitations"]], "", metadata["reload_check"], ""])
    (output / "report.md").write_text("\n".join(lines))
    print(audit.round(4).to_string(index=False))
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
