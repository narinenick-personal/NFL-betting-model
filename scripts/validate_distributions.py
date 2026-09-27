"""Collect rolling-origin residuals and validate joint score distributions."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_model.distribution_backtesting import DistributionConfig, run_distribution_backtest  # noqa: E402
from nfl_model.distribution_reporting import save_distribution_run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/game_modeling_dataset.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/distributions")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--first-season", type=int, default=2021)
    parser.add_argument("--audit-season", type=int, default=2025)
    parser.add_argument("--check-tests", action="store_true")
    args = parser.parse_args(argv)
    config = DistributionConfig(first_season=args.first_season, audit_season=args.audit_season, draws=args.draws, seed=args.seed)
    config.validate()
    tests = "not run by this command"
    if args.check_tests:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
        tests = result.stdout.strip().splitlines()[-1]
    if not args.dataset.exists():
        parser.error("Build the game modeling dataset first")
    games = pd.read_parquet(args.dataset)
    run = run_distribution_backtest(games, config, progress=lambda message: print(message, flush=True))
    path = save_distribution_run(run, args.output_dir, dataset_path=args.dataset, games=games, tests=tests)
    print("\nDevelopment-only family selection:")
    print(run.selection.round(4).to_string(index=False))
    print("\nSelected distribution audit diagnostics:")
    columns = ["model", "distribution", "target", "games", "crps", "coverage_50", "coverage_80", "coverage_95"]
    print(run.metrics.loc[run.metrics.phase.eq("retrospective_audit") & run.metrics.selected, columns].round(4).to_string(index=False))
    print(f"Tests: {tests}")
    print(f"Report: {path.resolve()}")


if __name__ == "__main__":
    main()
