"""Persist distribution diagnostics, calibration provenance and reloadable models."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_model.distribution_backtesting import DistributionRun


def distribution_report(run: DistributionRun) -> str:
    meta = run.metadata
    lines = ["# Joint score distribution evaluation", "",
             f"Out-of-time residual seasons: {meta['residual_seasons']}. Development evaluation: {meta['development_seasons']}. Retrospective audit: {meta['audit_season']}.",
             f"Draws per game and candidate: **{meta['draws_per_game']:,}**. Final residual pairs per model: {meta['final_calibration_rows']}.", "",
             "Each mean-model fold tunes only on earlier games. Residual distributions fit only earlier out-of-time prediction errors; development games select the family before audit evaluation.",
             "Bootstrap samples home/away residuals as intact pairs. Gaussian sampling uses an estimated two-score covariance with Ledoit–Wolf shrinkage. Both retain residual bias and censor scores below zero. Totals and margins are derived for every draw.", "",
             "## Development selection", "",
             "Lower joint energy score is better; it evaluates the home/away distribution together.", "",
             "| Mean model | Residual family | Games | Energy score | Selected |",
             "| --- | --- | ---: | ---: | --- |"]
    for row in run.selection.itertuples(index=False):
        lines.append(f"| {row.model} | {row.distribution} | {row.games} | {row.energy_score:.3f} | {'yes' if row.selected else ''} |")
    lines.extend(["", "## Retrospective audit — selected families", "",
                  "2025 was already inspected in baseline work. These results do not constitute a new untouched holdout.", "",
                  "| Mean model | Family | Energy score | Score correlation |",
                  "| --- | --- | ---: | ---: |"])
    for row in run.joint_metrics.loc[run.joint_metrics.phase.eq("retrospective_audit") & run.joint_metrics.selected].itertuples(index=False):
        lines.append(f"| {row.model} | {row.distribution} | {row.energy_score:.3f} | {row.mean_sample_correlation:.3f} |")
    lines.extend(["", "| Mean model | Target | CRPS | 50% coverage | 80% coverage | 95% coverage | 80% width |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    audit = run.metrics.loc[run.metrics.phase.eq("retrospective_audit") & run.metrics.selected]
    for row in audit.itertuples(index=False):
        lines.append(f"| {row.model} | {row.target} | {row.crps:.3f} | {row.coverage_50:.1%} | {row.coverage_80:.1%} | {row.coverage_95:.1%} | {row.width_80:.2f} |")
    lines.extend(["", "Coverage should be compared with its nominal percentage; CRPS, energy and interval scores are better when lower. Narrow intervals alone are not evidence of a good distribution.",
                  "The full metrics include 95% Wilson intervals around coverage, interval scores, bias and both candidate families. PIT histograms use randomized CDF ranks at atoms (including censored zero scores); roughly uniform bins are desirable, not a guarantee of calibration.", "",
                  "Energy scores use disjoint independent draw pairs for the pair-distance expectation. CRPS is exact for each finite empirical ensemble. Sampling error remains; the stored seed and per-game streams make results reproducible.", "",
                  "## Limits and next work", "", *[f"- {line}" for line in meta["limitations"]], "",
                  "Before pricing bets, develop a discrete scoring/settlement layer and validate moneyline, spread and total probabilities with chronological calibration, Brier score and log loss. Pushes and NFL tie/overtime rules need explicit treatment; rounding these draws is insufficient.", "",
                  "## Artifacts", "",
                  "- `residuals.parquet`: nested out-of-time predictions/errors; `calibration_eligible` is false for audit rows.",
                  "- `tuning.parquet`: alpha candidates and inner/outer fitting dates.",
                  "- `diagnostics.parquet` / `joint_diagnostics.parquet`: per-game distribution errors, intervals, PIT and joint scores.",
                  "- `metrics.parquet` / `joint_metrics.parquet` / `pit_histograms.parquet`: aggregate diagnostics.",
                  "- `selection.parquet`: development-only family selection.",
                  "- `models.joblib`: three fitted joint models, excluding audit outcomes from all fits.",
                  "- `example_simulations.parquet`: one audit game's complete draws from each selected model.",
                  "- `run.json`: protocol, versions, hashes, residual parameters and tests.", "",
                  f"Tests: {meta.get('tests', 'not run by this command')}.",
                  f"Reload check: {meta.get('reload_check', 'not yet recorded')}.", ""])
    return "\n".join(lines)


def save_distribution_run(run: DistributionRun, output: Path, *, dataset_path: Path, games: pd.DataFrame, tests: str) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = ("residuals", "tuning", "diagnostics", "joint_diagnostics", "metrics", "joint_metrics", "pit_histograms", "selection")
    for name in frames:
        getattr(run, name).to_parquet(output / f"{name}.parquet", index=False)
    joblib.dump(run.models, output / "models.joblib", compress=3)
    restored = joblib.load(output / "models.joblib")
    audit = games.loc[games.season.eq(run.metadata["audit_season"])].sort_values(["gameday", "game_id"])
    examples = []
    for name, model in restored.items():
        expected = run.diagnostics.loc[run.diagnostics.phase.eq("retrospective_audit") & run.diagnostics.model.eq(name) & run.diagnostics.selected]
        for i, (game_id, draws) in enumerate(model.simulate(audit, draws=run.metadata["draws_per_game"], seed=run.metadata["seed"])):
            target_means = expected.loc[expected.game_id.eq(game_id)].set_index("target").sample_mean
            if not np.allclose(draws.mean().loc[target_means.index], target_means, atol=1e-10, rtol=1e-10):
                raise ValueError(f"Reloaded simulation differs for {name}/{game_id}")
            if i == 0:
                draws = draws.assign(game_id=game_id, model=name, draw_id=np.arange(len(draws)))
                examples.append(draws)
    pd.concat(examples, ignore_index=True).to_parquet(output / "example_simulations.parquet", index=False)
    run.metadata.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "tests": tests,
        "reload_check": "All selected audit-game sample means reproduced with the saved seed",
        "dataset": {"path": str(dataset_path.resolve()), "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()},
        "versions": {pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py"))},
        "model_bundle_sha256": hashlib.sha256((output / "models.joblib").read_bytes()).hexdigest(),
        "final_parameters": {name: {"residual_bias": model.distribution.bias_.tolist(), "regularized_covariance": model.distribution.covariance_.tolist(),
                                    "shrinkage": model.distribution.shrinkage_, "calibration_end": str(model.distribution.calibration_end_), "mean_fit_end": str(model.mean_fit_end)} for name, model in run.models.items()},
        "metric_references": ["https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_ensemble.html", "https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.es_ensemble.html"],
    })
    (output / "run.json").write_text(json.dumps(run.metadata, indent=2, allow_nan=False) + "\n")
    path = output / "report.md"
    path.write_text(distribution_report(run))
    return path
