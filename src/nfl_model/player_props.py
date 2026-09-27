"""Explicit simulated player-stat contracts; no fabricated sportsbook quotes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_model.settlement import price_comparison

PROP_STATS = ("attempts", "carries", "targets", "receptions", "passing_yards", "rushing_yards",
              "receiving_yards", "passing_tds", "rushing_tds", "receiving_tds")


@dataclass(frozen=True)
class PropContract:
    statistic: str
    line: float
    side: str = "over"
    participation: str = "all_candidates"

    def validate(self):
        if self.statistic not in PROP_STATS or self.side not in ("over", "under"):
            raise ValueError("Unsupported player statistic or side")
        if not np.isfinite(self.line) or not float(self.line * 2).is_integer():
            raise ValueError("Player lines must be finite whole or half numbers")
        if self.participation not in ("all_candidates", "offense_snap_required"):
            raise ValueError("Unknown participation contract")


def prop_probabilities(values, contract: PropContract, *, offense_active=None) -> dict:
    contract.validate()
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values % 1 != 0).any():
        raise ValueError("Need finite integer player-stat draws")
    if not contract.statistic.endswith("yards") and (values < 0).any():
        raise ValueError("Count-statistic draws cannot be negative")
    action = np.ones(len(values), dtype=bool)
    if contract.participation == "offense_snap_required":
        activity = np.asarray(offense_active)
        if activity.shape != values.shape or not np.isin(activity, [0, 1]).all():
            raise ValueError("This contract requires aligned offensive-participation draws")
        action = activity.astype(bool)
    difference = values - contract.line
    if contract.side == "under":
        difference = -difference
    result = {"p_win": float((action & (difference > 0)).mean()), "p_loss": float((action & (difference < 0)).mean()),
              "p_push": float((action & (difference == 0)).mean()), "p_void": float((~action).mean()),
              "draws": len(values), "action_draws": int(action.sum())}
    decision = result["p_win"] + result["p_loss"]
    result["p_win_conditional"] = result["p_win"] / decision if decision else np.nan
    return result


def price_prop(values, contract: PropContract, odds: float, *, offense_active=None, opposite_odds=None) -> dict:
    probabilities = prop_probabilities(values, contract, offense_active=offense_active)
    comparison = price_comparison(probabilities["p_win"], probabilities["p_loss"], probabilities["p_push"] + probabilities["p_void"], odds, opposite_odds)
    return {**probabilities, **comparison, "statistic": contract.statistic, "line": contract.line,
            "side": contract.side, "participation": contract.participation, "odds": odds}
