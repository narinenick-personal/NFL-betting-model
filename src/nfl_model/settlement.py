"""Explicit full-game NFL settlement, using final scores inclusive of overtime."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSTSEASON_TYPES = ("WC", "DIV", "CON", "SB")


def overtime_format(season: int, game_type: str) -> dict:
    """Document the historical rule regime; this does not simulate OT possessions."""
    if season not in range(2021, 2026) or game_type not in ("REG", *POSTSEASON_TYPES):
        raise ValueError("Verified overtime formats cover 2021–2025 REG/WC/DIV/CON/SB only")
    postseason = game_type != "REG"
    both_after_td = season >= (2022 if postseason else 2025)
    return {
        "regime": f"{'POST' if postseason else 'REG'}_{'both_possessions' if both_after_td else 'opening_td_sudden_death'}",
        "period_minutes": 15 if postseason else 10,
        "final_tie_allowed": not postseason,
        "receiving_team_opening_td_ends_game": not both_after_td,
        "periods_until_winner": postseason,
        "note": "Both-possession opportunity is subject to rule exceptions and, in regular season, the 10-minute limit. Final-score model; no OT added after prediction.",
    }


@dataclass(frozen=True)
class SettlementRules:
    moneyline_tie: str = "push"
    includes_overtime: bool = True

    def validate(self) -> None:
        if self.moneyline_tie not in ("push", "loss"):
            raise ValueError("moneyline_tie must be 'push' or 'loss'")
        if not self.includes_overtime:
            raise ValueError("Only full-game markets including overtime are supported")


def validate_scores(home, away) -> tuple[np.ndarray, np.ndarray]:
    home, away = np.asarray(home, dtype=float), np.asarray(away, dtype=float)
    if home.shape != away.shape or not np.isfinite(home).all() or not np.isfinite(away).all():
        raise ValueError("Scores must have matching shapes and finite values")
    if (home < 0).any() or (away < 0).any() or (home % 1 != 0).any() or (away % 1 != 0).any():
        raise ValueError("Final scores must be nonnegative integers")
    return home, away


def grade_market(home, away, *, market: str, side: str, line: float | None = None, rules: SettlementRules = SettlementRules()) -> np.ndarray:
    """Return +1 win, 0 push, -1 loss. Spread line is the HOME handicap.

    Example: home -3 uses line=-3; home 24, away 21 pushes. The nflverse
    spread_line has the opposite sign and must be negated at the boundary.
    """
    rules.validate()
    home, away = validate_scores(home, away)
    if market == "moneyline":
        if side not in ("home", "away") or line is not None:
            raise ValueError("Moneyline requires a home/away side and no line")
        difference = home - away if side == "home" else away - home
    elif market in ("spread", "total"):
        if line is None or not np.isfinite(line) or not float(line * 2).is_integer():
            raise ValueError("Lines must be finite integers or half-points; quarter-lines are unsupported")
        if market == "spread":
            if side not in ("home", "away"):
                raise ValueError("Spread side must be home/away")
            difference = home - away + line
            if side == "away":
                difference = -difference
        else:
            if side not in ("over", "under") or line < 0:
                raise ValueError("Totals require over/under and a nonnegative line")
            difference = home + away - line
            if side == "under":
                difference = -difference
    else:
        raise ValueError(f"Unsupported market: {market}")
    result = np.sign(difference).astype(int)
    if market == "moneyline" and rules.moneyline_tie == "loss":
        result = np.where(result == 0, -1, result)
    return result


def american_profit(odds: float) -> float:
    if not np.isfinite(odds) or abs(odds) < 100:
        raise ValueError("American odds must be finite and <= -100 or >= +100")
    return float(odds / 100 if odds > 0 else 100 / -odds)


def implied_probability(odds: float) -> float:
    return 1 / (1 + american_profit(odds))


def no_vig_probability(odds: float, opposite_odds: float) -> float:
    ours, theirs = implied_probability(odds), implied_probability(opposite_odds)
    return ours / (ours + theirs)


def unit_profit(result, odds: float):
    """Net profit for a one-unit cash stake; a push returns the stake (net zero)."""
    result = np.asarray(result)
    if not np.isin(result, [-1, 0, 1]).all():
        raise ValueError("Settlement result must be -1, 0 or 1")
    return np.where(result == 1, american_profit(odds), np.where(result == -1, -1.0, 0.0))


def price_comparison(p_win: float, p_loss: float, p_push: float, odds: float, opposite_odds: float | None = None) -> dict:
    values = np.array([p_win, p_loss, p_push], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-10):
        raise ValueError("Win/loss/push probabilities must be nonnegative and sum to one")
    decision = p_win + p_loss
    conditional = p_win / decision if decision > 0 else np.nan
    implied = implied_probability(odds)
    return {
        "implied_probability": implied,
        "model_probability_conditional": conditional,
        "edge_vs_price_conditional": conditional - implied,
        "no_vig_probability": no_vig_probability(odds, opposite_odds) if opposite_odds is not None else np.nan,
        "expected_net_profit_per_unit": p_win * american_profit(odds) - p_loss,
    }
