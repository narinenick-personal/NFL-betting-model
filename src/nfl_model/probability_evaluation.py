"""Event probability scoring with pushes separated from decided outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data_quality import require_unique_keys


def event_metrics(rows: pd.DataFrame) -> dict:
    """Binary metrics condition on no push; multiclass metrics include pushes."""
    outcomes = rows.outcome.to_numpy(dtype=float)
    if not np.isin(outcomes, [-1, 0, 1]).all():
        raise ValueError("Unknown settlement outcome")
    probabilities = rows[["p_win", "p_loss", "p_push"]].to_numpy(dtype=float)
    supplied = np.isfinite(probabilities).all(axis=1)
    if (np.isfinite(probabilities).any(axis=1) & ~supplied).any() or np.isinf(probabilities).any():
        raise ValueError("Supply all unconditional probabilities or leave all three missing")
    if ((probabilities[supplied] < 0) | (probabilities[supplied] > 1)).any() or not np.allclose(probabilities[supplied].sum(axis=1), 1):
        raise ValueError("Invalid win/loss/push probabilities")
    q = rows.p_conditional.to_numpy(dtype=float)
    finite = np.isfinite(q)
    if np.isinf(q).any() or ((q[finite] < 0) | (q[finite] > 1)).any():
        raise ValueError("Invalid conditional win probability")
    determined = supplied & (probabilities[:, :2].sum(axis=1) > 0)
    if (supplied & ~determined & ~np.isnan(q)).any():
        raise ValueError("Conditional probability is undefined when all mass is a push")
    if not np.allclose(q[determined], probabilities[determined, 0] / probabilities[determined, :2].sum(axis=1)):
        raise ValueError("Conditional win probability disagrees with win/loss mass")
    decisions = (outcomes != 0) & finite
    y = (outcomes[decisions] == 1).astype(float)
    clipped = np.clip(q[decisions], 1e-15, 1 - 1e-15)
    result = {
        "games": len(rows), "decisions": int(decisions.sum()), "pushes": int((outcomes == 0).sum()),
        "binary_brier": float(np.mean((q[decisions] - y)**2)) if len(y) else np.nan,
        "binary_log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))) if len(y) else np.nan,
        "unconditional_games": int(supplied.sum()),
        "observed_push_rate": float((outcomes == 0).mean()),
        "mean_predicted_push_rate": float(probabilities[supplied, 2].mean()) if supplied.any() else np.nan,
        "push_brier": float(np.mean((probabilities[supplied, 2] - (outcomes[supplied] == 0))**2)) if supplied.any() else np.nan,
    }
    if supplied.any():
        # Classes in p-column order: win, loss, push.
        labels = np.where(outcomes[supplied] == 1, 0, np.where(outcomes[supplied] == -1, 1, 2))
        truth = np.eye(3)[labels]
        result["multiclass_brier"] = float(np.square(probabilities[supplied] - truth).sum(axis=1).mean())
        result["multiclass_log_loss"] = float(-np.log(np.maximum(probabilities[supplied][np.arange(len(labels)), labels], 1e-15)).mean())
    else:
        result.update(multiclass_brier=np.nan, multiclass_log_loss=np.nan)
    return result


def summarize_probabilities(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_unique_keys(predictions, ["game_id", "model", "market", "side"], "market predictions")
    primary = predictions.loc[predictions.primary_side]
    keys = ["phase", "season", "model", "market"]
    summaries, bins = [], []
    for identity, rows in primary.groupby(keys, sort=False):
        values = dict(zip(keys, identity))
        summaries.append({**values, **event_metrics(rows)})
        decided = rows.loc[rows.outcome.ne(0) & rows.p_conditional.notna()].copy()
        decided["bin"] = np.minimum((decided.p_conditional * 10).astype(int), 9)
        for bucket, part in decided.groupby("bin"):
            bins.append({**values, "bin_lower": bucket / 10, "bin_upper": (bucket + 1) / 10,
                         "games": len(part), "mean_probability": float(part.p_conditional.mean()), "observed_win_rate": float(part.outcome.eq(1).mean())})
    return pd.DataFrame(summaries), pd.DataFrame(bins)
