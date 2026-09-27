"""Build historical NFL team/game datasets. Defaults to the 2021–2025 seasons."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_model.data_loader import DEFAULT_CACHE_DIR, load_schedules, load_team_stats  # noqa: E402
from nfl_model.modeling_dataset import build_modeling_dataset  # noqa: E402
from nfl_model.reporting import dataset_manifest, dataset_report, format_report  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2021, 2026)))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--refresh", action="store_true", help="Refresh the local raw parquet caches")
    parser.add_argument("--check-tests", action="store_true", help="Run pytest before building; fail on any test failure")
    args = parser.parse_args(argv)
    seasons = sorted(set(args.seasons))
    tests = "not run by this build (run python -m pytest -q or use --check-tests)"
    if args.check_tests:
        check = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        print(check.stdout, end="", flush=True)
        if check.returncode:
            print(check.stderr, file=sys.stderr)
            raise SystemExit(check.returncode)
        tests = check.stdout.strip().splitlines()[-1]

    print(f"Loading schedules and team stats for {seasons}...", flush=True)
    schedules = load_schedules(seasons, refresh=args.refresh)
    stats = load_team_stats(seasons, refresh=args.refresh)
    windows = (3, 5)
    dataset = build_modeling_dataset(stats, schedules, windows=windows)
    actual_seasons = set(dataset.games["season"].unique())
    if actual_seasons != set(seasons):
        raise ValueError(f"Requested seasons {seasons} do not match completed-game coverage {sorted(actual_seasons)}")
    report = dataset_report(dataset, tests=tests)
    report["excluded_schedule_rows"] = len(schedules) - len(dataset.games)
    manifest = dataset_manifest(dataset, windows)
    manifest["created_at_utc"] = report["created_at_utc"]
    manifest["versions"] = {package: version(package) for package in ("nflreadpy", "pandas", "numpy", "pyarrow")}
    manifest["source_files"] = {}
    suffix = "_".join(map(str, seasons))
    for name in ("schedules", "team_stats"):
        cache = DEFAULT_CACHE_DIR / f"{name}_{suffix}.parquet"
        manifest["source_files"][name] = {"path": str(cache), "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset.games.to_parquet(output / "game_modeling_dataset.parquet", index=False)
    dataset.team_games.to_parquet(output / "team_game_dataset.parquet", index=False)
    for filename, payload in (("game_modeling_manifest.json", manifest), ("game_modeling_report.json", report)):
        (output / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    summary = format_report(dataset, report)
    (output / "game_modeling_report.txt").write_text(summary + "\n")
    print(summary)
    print(f"Saved datasets, manifest, and reports to {output}")


if __name__ == "__main__":
    main()
