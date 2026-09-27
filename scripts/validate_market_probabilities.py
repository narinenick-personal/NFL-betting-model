"""Fit discrete score grids and audit full-game market probabilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_model.market_backtesting import run_market_backtest  # noqa: E402
from nfl_model.market_reporting import file_hash, save_market_run  # noqa: E402
from nfl_model.settlement import SettlementRules  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/game_modeling_dataset.parquet")
    parser.add_argument("--distribution-dir", type=Path, default=ROOT / "artifacts/distributions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/markets")
    parser.add_argument("--moneyline-tie", choices=("push", "loss"), default="push")
    parser.add_argument("--check-tests", action="store_true")
    args = parser.parse_args(argv)
    source_paths = {name: args.distribution_dir / filename for name, filename in (
        ("residuals", "residuals.parquet"), ("mean_models", "models.joblib"), ("distribution_run", "run.json"))}
    for path in [args.dataset, *source_paths.values()]:
        if not path.is_file():
            parser.error(f"Missing {path}; build the dataset and validate_distributions first")
    manifest = json.loads(source_paths["distribution_run"].read_text())
    if file_hash(args.dataset) != manifest["dataset"]["sha256"]:
        parser.error("Dataset differs from the cached distribution run; regenerate distribution artifacts")
    if file_hash(source_paths["mean_models"]) != manifest["model_bundle_sha256"]:
        parser.error("Cached model bundle fails its recorded hash check")
    if not manifest["first_season"] == 2021 or not manifest["audit_season"] == 2025:
        parser.error("This settlement experiment verifies the 2021–2025 rule window only")
    tests = "not run by this command"
    if args.check_tests:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
        tests = result.stdout.strip().splitlines()[-1]
    games = pd.read_parquet(args.dataset)
    run = run_market_backtest(games, pd.read_parquet(source_paths["residuals"]),
                              mean_models=joblib.load(source_paths["mean_models"]),
                              rules=SettlementRules(moneyline_tie=args.moneyline_tie),
                              progress=lambda message: print(message, flush=True))
    report = save_market_run(run, args.output_dir, dataset_path=args.dataset, games=games,
                             source_paths=source_paths, tests=tests)
    columns = ["model", "market", "games", "decisions", "binary_brier", "binary_log_loss", "observed_push_rate", "mean_predicted_push_rate"]
    print(run.metrics.loc[run.metrics.phase.eq("retrospective_audit"), columns].round(5).to_string(index=False))
    print(f"Tests: {tests}")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()
