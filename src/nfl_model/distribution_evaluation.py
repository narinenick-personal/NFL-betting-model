"""Proper scoring rules, interval coverage and PIT diagnostics for score draws."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.team_games import TARGET_COLUMNS

LEVELS = (0.50, 0.80, 0.95)


def ensemble_crps(actual: float, samples: np.ndarray) -> float:
    """Exact empirical-ensemble CRPS in O(M log M), in outcome units."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(actual):
        raise ValueError("CRPS requires finite univariate draws and an observation")
    ordered = np.sort(values)
    n = len(values)
    spread = np.dot(2 * np.arange(1, n + 1) - n - 1, ordered) / n**2
    return float(np.abs(values - actual).mean() - spread)


def energy_score(actual: np.ndarray, samples: np.ndarray) -> float:
    """Monte Carlo energy score using independent disjoint draw pairs.

    Estimates E||X-y|| - 0.5 E||X-X'|| without an M-squared distance matrix.
    This is an unbiased estimator for IID draws from the underlying distribution,
    not the exact empirical-ensemble energy score. At least two draws are needed.
    """
    actual, values = np.asarray(actual, dtype=float), np.asarray(samples, dtype=float)
    if values.ndim != 2 or len(values) < 2 or actual.shape != (values.shape[1],):
        raise ValueError("Energy score requires matching multivariate observations and draws")
    if not np.isfinite(actual).all() or not np.isfinite(values).all():
        raise ValueError("Energy score requires finite values")
    pairs = len(values) // 2
    return float(np.linalg.norm(values - actual, axis=1).mean() - 0.5 * np.linalg.norm(values[:pairs] - values[pairs:2 * pairs], axis=1).mean())


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("Invalid binomial counts")
    p, denominator = successes / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def evaluate_draws(actual, samples: np.ndarray, *, pit_rng: np.random.Generator) -> tuple[list[dict], dict]:
    values = np.asarray(samples, dtype=float)
    observed = np.asarray(actual, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or observed.shape != (4,) or len(values) < 2:
        raise ValueError("Expected four coherent outcomes and (n, 4) draws")
    if not np.isfinite(values).all() or not np.isfinite(observed).all():
        raise ValueError("Draws and outcomes must be finite")
    if not np.allclose(values[:, 2], values[:, :2].sum(axis=1)) or not np.allclose(values[:, 3], values[:, 0] - values[:, 1]):
        raise ValueError("Total and margin must reconcile with sampled team scores")
    if not np.allclose(observed[2:], [observed[0] + observed[1], observed[0] - observed[1]]) or (values[:, :2] < 0).any():
        raise ValueError("Invalid actual outcomes or negative simulated scores")
    marginal = []
    for i, target in enumerate(TARGET_COLUMNS):
        draws, outcome = values[:, i], observed[i]
        row = {
            "target": target, "actual": outcome, "sample_mean": float(draws.mean()),
            "sample_std": float(draws.std(ddof=1)), "crps": ensemble_crps(outcome, draws),
            # Randomization handles atoms, notably the zero-censoring mass.
            "pit": float(((draws < outcome).sum() + pit_rng.uniform() * (draws == outcome).sum()) / len(draws)),
        }
        for level in LEVELS:
            alpha = 1 - level
            lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
            suffix = str(round(100 * level))
            row.update({
                f"lower_{suffix}": float(lower), f"upper_{suffix}": float(upper),
                f"covered_{suffix}": bool(lower <= outcome <= upper),
                f"width_{suffix}": float(upper - lower),
                f"interval_score_{suffix}": float(upper - lower + 2 / alpha * max(lower - outcome, 0) + 2 / alpha * max(outcome - upper, 0)),
            })
        marginal.append(row)
    covariance = np.cov(values[:, :2], rowvar=False)
    scale = float(np.sqrt(covariance[0, 0] * covariance[1, 1]))
    joint = {
        "energy_score": energy_score(observed[:2], values[:, :2]),
        "sample_correlation": float(covariance[0, 1] / scale) if scale > 0 else np.nan,
        "home_zero_mass": float((values[:, 0] == 0).mean()),
        "away_zero_mass": float((values[:, 1] == 0).mean()),
    }
    return marginal, joint


def summarize_distributions(marginals: pd.DataFrame, joints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["phase", "season", "model", "distribution"]
    marginal_rows, pit_rows = [], []
    for identity, group in marginals.groupby([*keys, "target"], sort=False):
        row = dict(zip([*keys, "target"], identity))
        row.update(games=len(group), crps=float(group.crps.mean()), mean_prediction_bias=float((group.sample_mean - group.actual).mean()))
        for level in LEVELS:
            suffix = str(round(level * 100))
            successes = int(group[f"covered_{suffix}"].sum())
            low, high = wilson_interval(successes, len(group))
            row.update({f"coverage_{suffix}": successes / len(group), f"coverage_{suffix}_ci_low": low, f"coverage_{suffix}_ci_high": high,
                        f"width_{suffix}": float(group[f"width_{suffix}"].mean()), f"interval_score_{suffix}": float(group[f"interval_score_{suffix}"].mean())})
        marginal_rows.append(row)
        counts, edges = np.histogram(group.pit, bins=np.linspace(0, 1, 11))
        for i, count in enumerate(counts):
            pit_rows.append({**dict(zip([*keys, "target"], identity)), "bin_lower": edges[i], "bin_upper": edges[i + 1], "count": int(count), "expected_uniform_count": len(group) / 10})
    joint_summary = joints.groupby(keys, sort=False).agg(
        games=("game_id", "size"), energy_score=("energy_score", "mean"),
        mean_sample_correlation=("sample_correlation", "mean"), home_zero_mass=("home_zero_mass", "mean"), away_zero_mass=("away_zero_mass", "mean"),
    ).reset_index()
    return pd.DataFrame(marginal_rows), joint_summary, pd.DataFrame(pit_rows)
