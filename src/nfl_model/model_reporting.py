"""Persist reproducible baseline artifacts and readable evaluation reports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np

from nfl_model.backtesting import BaselineRun
from nfl_model.team_games import TARGET_COLUMNS


def format_baseline_report(run: BaselineRun) -> str:
    meta = run.metadata
    lines = [
        "# NFL score baseline evaluation", "",
        f"Training seasons: {meta['train_seasons']}; validation: {meta['validation_season']}; held-out test: {meta['test_season']}.",
        f"Game counts: {meta['split_rows']}. Final models refit on {meta['final_fit_rows']} games through {meta['validation_season']}.", "",
        f"Football inputs: {len(meta['football_features'])}; optional market inputs: {len(meta['market_features'])}.",
        f"Selected alphas: {meta['selected_alphas']}. Selection used {meta['selection_metric']} only.",
        "All preprocessing is fitted within each training sample. Score outputs are continuous means; totals and margins reconcile with the two team scores.", "",
    ]
    for split in ("validation", "test"):
        lines.extend([f"## {split.title()} RMSE (points, common games)", "",
                      "| Model | Home score | Away score | Game total | Home margin |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        data = run.metrics.loc[run.metrics.split.eq(split) & run.metrics.cohort.eq("common")]
        for name in run.models:
            model_metrics = data.loc[data.model.eq(name)].set_index("target")
            values = [f"{model_metrics.loc[target, 'rmse']:.3f}" for target in TARGET_COLUMNS]
            lines.append("| " + " | ".join([name, *values]) + " |")
        counts = data.groupby("target").scored_games.first().to_dict()
        lines.extend(["", f"Common-game counts by target: {counts}.", ""])
    lines.extend(["## Test MAE and bias (points, common games)", "",
                  "| Model | Target | MAE | Bias (prediction − actual) | Games |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for row in run.metrics.loc[run.metrics.split.eq("test") & run.metrics.cohort.eq("common")].itertuples(index=False):
        lines.append(f"| {row.model} | {row.target} | {row.mae:.3f} | {row.bias:.3f} | {row.scored_games} |")
    lines.extend(["", "## Coverage and interpretation", "",
                  "Common comparisons use the same games for all models within each target. The full metrics artifact also reports available-case errors and coverage for each model. Missing market lines are not filled for the direct market benchmark.", "",
                  "The training-mean benchmark estimates separate home/away averages. The recent-team benchmark averages a team's last-five scoring average with its opponent's last-five points-allowed average; missing components use training-set home/away means.", "",
                  "The direct market benchmark uses total_line as total and spread_line as home margin; its implied team scores are (total ± margin) / 2. This is a line-based benchmark, not a claim that prices equal statistical means.", "",
                  "Validation results helped choose regularization. Final models include 2024 only after selection. The fixed model coefficients do not update during the test season; pregame features advance using already completed games.", "",
                  *[f"- {item}" for item in meta["limitations"]], "",
                  "## Validation", "", f"Tests: {meta.get('tests', 'not run by this training command')}.",
                  "Predictions, metrics, tuning scores, fitted pipelines and provenance are saved beside this report.", ""])
    return "\n".join(lines)


def save_baseline_run(
    run: BaselineRun, output_dir: Path, *, dataset_path: Path, tests: str = "not run by this training command"
) -> Path:
    output_dir = Path(output_dir)
    dataset_path = Path(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    run.metadata.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests": tests,
        "dataset": {"path": str(dataset_path.resolve()), "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()},
        "versions": {package: version(package) for package in ("scikit-learn", "numpy", "pandas", "scipy", "joblib")},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py"))},
    })
    run.predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    run.metrics.to_parquet(output_dir / "metrics.parquet", index=False)
    run.tuning.to_parquet(output_dir / "tuning.parquet", index=False)
    joblib.dump(run.models, output_dir / "models.joblib", compress=3)
    run.metadata["model_bundle_sha256"] = hashlib.sha256((output_dir / "models.joblib").read_bytes()).hexdigest()
    (output_dir / "run.json").write_text(json.dumps(run.metadata, indent=2, allow_nan=False) + "\n")
    report_path = output_dir / "report.md"
    report_path.write_text(format_baseline_report(run))
    return report_path


def verify_saved_predictions(run: BaselineRun, games, output_dir: Path) -> None:
    """Confirm this freshly saved local bundle reproduces every test prediction."""
    models = joblib.load(Path(output_dir) / "models.joblib")
    test = games.loc[games.season.eq(run.metadata["test_season"])].copy()
    for name, model in models.items():
        expected = run.predictions.loc[run.predictions.split.eq("test") & run.predictions.model.eq(name)].set_index("game_id")
        actual = model.predict(test)
        for target in TARGET_COLUMNS:
            if not np.allclose(actual[target], expected.loc[test.game_id, f"pred_{target}"], equal_nan=True, rtol=1e-10, atol=1e-10):
                raise ValueError(f"Reloaded model predictions differ for {name}/{target}")
