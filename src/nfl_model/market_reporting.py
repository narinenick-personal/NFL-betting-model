"""Save exact market probabilities, simulation checks, and their provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_model.distributions import game_rng
from nfl_model.market_backtesting import MarketProbabilityRun, price_game
from nfl_model.settlement import SettlementRules, grade_market


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def market_report(run: MarketProbabilityRun) -> str:
    meta = run.metadata
    lines = ["# Discrete scores and market probability validation", "",
             f"Development: {meta['first_evaluation_season']}–{meta['audit_season'] - 1}. Retrospective audit: {meta['audit_season']}.",
             "All score-frequency corrections, covariance estimates, and tie rates use earlier games only. Mean forecasts are reused from nested chronological folds. The grid uses a fixed Gaussian kernel; no distribution or parameter selection uses this audit.", "",
             "Each grid represents joint integer final scores including overtime. Market probabilities sum its mass exactly. Simulations sample from the fitted grid; they do not create or fit predictions.",
             f"Settlement: full game, moneyline tie = **{meta['moneyline_tie']}**. Spread `line` is always the **home handicap**, including in away-side rows. Total `line` is the total threshold.", "",
             "## Retrospective audit", "",
             "2025 has already been inspected. Lower Brier score and log loss are better. Binary metrics condition on no push and use one canonical side per game: home moneyline, home spread, over total.", "",
             "| Market | Model | Games | Decisions | Brier | Log loss | Observed push | Predicted push |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    audit = run.metrics.loc[run.metrics.phase.eq("retrospective_audit")].sort_values(["market", "model"])
    for row in audit.itertuples(index=False):
        push = f"{row.mean_predicted_push_rate:.2%}" if pd.notna(row.mean_predicted_push_rate) else "not inferred"
        lines.append(f"| {row.market} | {row.model} | {row.games} | {row.decisions} | {row.binary_brier:.4f} | {row.binary_log_loss:.4f} | {row.observed_push_rate:.2%} | {push} |")
    lines.extend(["", "The odds benchmark proportionally removes the overround from the two prices. It does not identify push probability, so its unconditional and three-class metrics remain missing. Missing prices can reduce its cohort; compare game/decision counts before comparing scores.", "",
                  "`metrics.parquet` also contains win/loss/push three-class Brier scores (sum of three squared errors, range 0–2), three-class log loss, and push-only Brier scores. `calibration_bins.parquet` contains counts, mean forecast and observed win rate in ten fixed probability bins, excluding actual pushes. Small bins are descriptive, not evidence of calibration.", "",
                  "## Fitting boundaries", "",
                  "| Forecast season | Model | Residual games | Score-history games | Latest score history | Tie prior |",
                  "| --- | --- | ---: | ---: | --- | ---: |"])
    for row in run.fit_log.itertuples(index=False):
        lines.append(f"| {row.season} | {row.model} | {row.calibration_games} | {row.score_history_games} | {pd.Timestamp(row.score_history_end).date()} | {row.tie_prior_rate:.3%} |")
    lines.extend(["", "The score grid conditions on nonnegative supported scores; it does not round or censor continuous draws. Local score and symmetric margin corrections retain historical scoring spikes. Tie odds receive a separate pooled historical adjustment; playoff ties have zero terminal mass. Grid weighting changes moments, so the final grid mean need not equal the input regression mean.", "",
                  "## Price comparison", "",
                  "For a one-unit cash stake with win profit b, expected net profit is `p_win * b - p_loss`; a push returns the stake. `edge_vs_price_conditional` subtracts the quoted break-even probability from `p_win / (p_win + p_loss)`. This is a diagnostic of the supplied snapshot, not a selected betting strategy or measured return.", "",
                  "## Limitations", "", *[f"- {item}" for item in meta["limitations"]], "",
                  "## Artifacts and checks", "",
                  "- `predictions.parquet`: both sides of all supported markets, probabilities, observed grades, odds, edge and expected profit.",
                  "- `metrics.parquet` / `calibration_bins.parquet`: development and audit probability diagnostics.",
                  "- `fit_log.parquet`: fold cutoffs, sample sizes and tie-rate parameters.",
                  "- `models.joblib`: discrete game models fitted before the audit.",
                  "- `simulation_checks.parquet`: sampled versus exact win/loss/push rates for canonical markets in every audit game.",
                  f"- `example_simulations.parquet`: {meta.get('draws_per_audit_game', 10_000):,} complete integer-score draws for one audit game per model.",
                  "- `run.json`: versions, input/source/artifact hashes, fixed assumptions and verification results.", "",
                  f"Tests: {meta.get('tests', 'not run by this command')}.",
                  f"Reload check: {meta.get('reload_check', 'not recorded')}.", "",
                  "## Rule references", "", *[f"- {url}" for url in meta["rule_sources"]], ""])
    return "\n".join(lines)


def save_market_run(run: MarketProbabilityRun, output: Path, *, dataset_path: Path,
                    games: pd.DataFrame, source_paths: dict[str, Path], tests: str,
                    draws: int = 10_000, seed: int = 20260925) -> Path:
    if not run.models:
        raise ValueError("Saved mean models are required to persist a reproducible run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "metrics", "calibration_bins", "fit_log"):
        getattr(run, name).to_parquet(output / f"{name}.parquet", index=False)
    bundle = output / "models.joblib"
    joblib.dump(run.models, bundle, compress=3)
    restored = joblib.load(bundle)
    audit = games.loc[games.season.eq(run.metadata["audit_season"])].sort_values(["gameday", "game_id"])
    rows = {row.game_id: row for row in audit.itertuples(index=False)}
    expected = run.predictions.loc[run.predictions.phase.eq("retrospective_audit")].set_index(["model", "game_id", "market", "side"])
    rules = SettlementRules(moneyline_tie=run.metadata["moneyline_tie"])
    examples, checks = [], []
    for name, model in restored.items():
        for i, (game_id, grid) in enumerate(model.grids(audit)):
            priced = price_game(rows[game_id], grid, model=name, phase="retrospective_audit", rules=rules)
            samples = grid.sample(draws=draws, rng=game_rng(seed, game_id, "discrete_scores"))
            if i == 0:
                examples.append(samples.assign(game_id=game_id, model=name, draw_id=np.arange(draws)))
            for row in priced:
                original = expected.loc[(name, game_id, row["market"], row["side"])]
                columns = ["p_win", "p_loss", "p_push"]
                if not np.allclose([row[c] for c in columns], original[columns].to_numpy(dtype=float), atol=1e-12, rtol=1e-12):
                    raise ValueError(f"Reloaded probabilities disagree for {name}/{game_id}")
                if row["primary_side"]:
                    grades = grade_market(samples.home_score, samples.away_score, market=row["market"], side=row["side"], line=row["line"], rules=rules)
                    checks.append({"model": name, "game_id": game_id, "market": row["market"], "side": row["side"], "draws": draws,
                                   **{f"exact_{label}": row[f"p_{label}"] for label in ("win", "loss", "push")},
                                   **{f"sample_{label}": float((grades == value).mean()) for label, value in (("win", 1), ("loss", -1), ("push", 0))}})
    pd.concat(examples, ignore_index=True).to_parquet(output / "example_simulations.parquet", index=False)
    pd.DataFrame(checks).to_parquet(output / "simulation_checks.parquet", index=False)
    run.metadata.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "tests": tests,
        "draws_per_audit_game": draws, "seed": seed,
        "reload_check": "All audit-game market probabilities reproduced after model reload; integer simulations generated for every audit game",
        "dataset": {"path": str(dataset_path.resolve()), "sha256": file_hash(dataset_path)},
        "inputs": {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in source_paths.items()},
        "versions": {pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
        "source_sha256": {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "artifact_sha256": {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.suffix in (".parquet", ".joblib")},
    })
    (output / "run.json").write_text(json.dumps(run.metadata, indent=2, allow_nan=False) + "\n")
    path = output / "report.md"
    path.write_text(market_report(run))
    return path
