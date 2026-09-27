"""Train score baselines, choose regularization on validation, and score the holdout."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_model.backtesting import BacktestConfig, DEFAULT_ALPHAS, run_baseline_backtest  # noqa: E402
from nfl_model.model_reporting import save_baseline_run, verify_saved_predictions  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "processed" / "game_modeling_dataset.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "baselines")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--check-tests", action="store_true")
    args = parser.parse_args(argv)
    config = BacktestConfig(tuple(args.train_seasons), args.validation_season, args.test_season, tuple(args.alphas))
    config.validate()
    tests = "not run by this training command"
    if args.check_tests:
        check = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        print(check.stdout, end="", flush=True)
        if check.returncode:
            print(check.stderr, file=sys.stderr)
            raise SystemExit(check.returncode)
        tests = check.stdout.strip().splitlines()[-1]
    if not args.dataset.exists():
        parser.error(f"Dataset not found: {args.dataset}. Run scripts/build_modeling_dataset.py first.")
    games = pd.read_parquet(args.dataset)
    print(f"Training baselines: {config.train_seasons}; validating {config.validation_season}; testing {config.test_season}...", flush=True)
    run = run_baseline_backtest(games, config)
    report_path = save_baseline_run(run, args.output_dir, dataset_path=args.dataset, tests=tests)
    verify_saved_predictions(run, games, args.output_dir)
    print(f"Selected alphas: {run.metadata['selected_alphas']}")
    comparison = run.metrics.loc[run.metrics.cohort.eq("common"), ["split", "model", "target", "scored_games", "mae", "rmse", "bias"]]
    print(comparison.round(3).to_string(index=False))
    print(f"Tests: {tests}")
    print("Saved model reload check passed.")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
