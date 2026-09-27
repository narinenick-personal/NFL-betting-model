"""Regression evaluation with explicit sample coverage and common-game comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data_quality import require_unique_keys
from nfl_model.team_games import TARGET_COLUMNS


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Bias is prediction minus actual; no silent filtering of invalid values."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics require nonempty matching shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite observations and predictions")
    error = predicted - actual
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean())}


def evaluation_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Available-case and common-game metrics for each split/model/target.

    The common cohort is target-specific and excludes any game without a
    prediction from every comparator. Missing market lines are never imputed.
    """
    require_unique_keys(predictions, ["split", "model", "game_id"], "predictions")
    rows = []
    for split, split_rows in predictions.groupby("split", sort=False):
        model_count = split_rows.model.nunique()
        for target in TARGET_COLUMNS:
            pred = f"pred_{target}"
            actual = split_rows[target].to_numpy(dtype=float)
            estimates = split_rows[pred].to_numpy(dtype=float)
            if not np.isfinite(actual).all() or np.isinf(estimates).any():
                raise ValueError("Evaluation received invalid targets or infinite predictions")
            if split_rows.groupby("game_id")[target].nunique().ne(1).any():
                raise ValueError("Comparator targets disagree for the same game")
            available_counts = split_rows.loc[split_rows[pred].notna()].groupby("game_id").model.nunique()
            common_ids = available_counts.index[available_counts.eq(model_count)]
            for model, model_rows in split_rows.groupby("model", sort=False):
                for cohort in ("available", "common"):
                    mask = model_rows[pred].notna()
                    if cohort == "common":
                        mask &= model_rows.game_id.isin(common_ids)
                    selected = model_rows.loc[mask]
                    metrics = regression_metrics(selected[target], selected[pred]) if len(selected) else {"mae": np.nan, "rmse": np.nan, "bias": np.nan}
                    rows.append({
                        "split": split, "model": model, "target": target, "cohort": cohort,
                        "games": len(model_rows), "scored_games": len(selected),
                        "coverage_pct": 100 * len(selected) / len(model_rows), **metrics,
                    })
    return pd.DataFrame(rows)
