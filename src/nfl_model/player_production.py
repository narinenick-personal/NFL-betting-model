"""Player efficiency estimators and out-of-time dispersion calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression, Ridge

from nfl_model.opportunity_models import OPPORTUNITIES, player_inputs, preprocessing
from nfl_model.player_dataset import PLAYER_FEATURES


@dataclass
class PlayerProductionModel:
    """Weighted per-opportunity efficiencies; no current-game exposure as input."""
    alpha: float = 1000.

    def fit(self, rows: pd.DataFrame) -> PlayerProductionModel:
        self.fit_end_ = pd.to_datetime(rows.gameday).max()
        self.preprocessor_ = preprocessing(PLAYER_FEATURES, ("position",))
        X = self.preprocessor_.fit_transform(player_inputs(rows))
        targets = rows.targets.to_numpy(dtype=float)
        receptions = rows.receptions.to_numpy(dtype=float)
        eligible = targets > 0
        if not eligible.any() or (receptions > targets).any():
            raise ValueError("Catch model needs valid receiving exposure")
        # Aggregate binomial likelihood: successes/failures with count weights.
        catch_X = np.concatenate([X[eligible], X[eligible]])
        catch_y = np.concatenate([np.ones(eligible.sum()), np.zeros(eligible.sum())])
        catch_weights = np.concatenate([receptions[eligible], (targets - receptions)[eligible]])
        keep = catch_weights > 0
        self.catch_ = LogisticRegression(C=.05, max_iter=500).fit(catch_X[keep], catch_y[keep], sample_weight=catch_weights[keep])
        self.yards_ = {}
        for name, count, total in (("receiving", "receptions", "receiving_yards"), ("rushing", "carries", "rushing_yards")):
            mask = rows[count].gt(0)
            n = rows.loc[mask, count].to_numpy(dtype=float)
            if not len(n):
                raise ValueError(f"No {name} exposure")
            self.yards_[name] = Ridge(alpha=self.alpha, solver="lsqr", tol=1e-8).fit(X[mask], rows.loc[mask, total].to_numpy() / n, sample_weight=n)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = self.preprocessor_.transform(player_inputs(rows))
        return pd.DataFrame({"catch_probability": np.clip(self.catch_.predict_proba(X)[:, 1], .001, .999),
                             "receiving_yards_per_catch": np.clip(self.yards_["receiving"].predict(X), -99, 99),
                             "rushing_yards_per_carry": np.clip(self.yards_["rushing"].predict(X), -99, 99)}, index=rows.index)


def production_calibration(rows: pd.DataFrame) -> dict:
    """Moment estimates from earlier OUT-OF-TIME predictions, including shared errors."""
    if rows.empty or not pd.to_datetime(rows.fit_end).lt(pd.to_datetime(rows.gameday)).all():
        raise ValueError("Production calibration needs strictly out-of-time predictions")
    result = {"calibration_end": pd.to_datetime(rows.gameday).max(), "rows": len(rows)}
    n, k, p = (rows[c].to_numpy(dtype=float) for c in ("targets", "receptions", "catch_probability"))
    denominator = np.sum(n * (n - 1) * p * (1 - p))
    rho = np.sum((k - n * p)**2 - n * p * (1 - p)) / denominator if denominator > 0 else 0
    result["catch_intraclass_correlation"] = float(np.clip(rho, 0, .5))
    for name, count, target, forecast in (("receiving", "receptions", "receiving_yards", "receiving_yards_per_catch"),
                                           ("rushing", "carries", "rushing_yards", "rushing_yards_per_carry")):
        selected = rows.loc[rows[count].gt(0)].copy()
        if selected.empty:
            raise ValueError("Missing yardage calibration exposure")
        n = selected[count].to_numpy(dtype=float)
        error = selected[target].to_numpy(dtype=float) - n * selected[forecast].to_numpy(dtype=float)
        bias = float(error.sum() / n.sum())
        error -= n * bias
        moments = selected[["game_id", "team"]].assign(error=error, error2=error**2, n=n, n2=n**2).groupby(["game_id", "team"]).sum()
        pair_denominator = (moments.n**2 - moments.n2).sum()
        shared = float(max(0, (moments.error**2 - moments.error2).sum() / pair_denominator)) if pair_denominator > 0 else 0.
        individual = float(max(1e-6, np.sum(error**2 - n**2 * shared) / n.sum()))
        result[name] = {"bias_per_opportunity": bias, "shared_variance": shared, "individual_variance": individual}
    return result


def fit_allocation_concentrations(calibration_rows: pd.DataFrame) -> dict:
    """Dirichlet-multinomial MLE conditional on observed earlier participation.

    Rows contain earlier OOT conditional weights. Unknown-participation groups
    are skipped; this conditioning estimates share dispersion, not availability.
    """
    if not pd.to_datetime(calibration_rows.fit_end).lt(pd.to_datetime(calibration_rows.gameday)).all():
        raise ValueError("Share dispersion requires out-of-time weights")
    result = {}
    groups = list(calibration_rows.groupby(["game_id", "team"], sort=False))
    for name in OPPORTUNITIES:
        samples = []
        for _, group in groups:
            if group.offense_active.isna().any():
                continue
            active = group.loc[group.offense_active.eq(1) | group.is_other.eq(1)]
            counts = active[name].to_numpy(dtype=float)
            weights = active[f"weight_{name}"].to_numpy(dtype=float)
            if counts.sum() <= 0 or weights.sum() <= 0 or len(counts) < 2:
                continue
            q = np.maximum(weights / weights.sum(), 1e-9)
            q /= q.sum()
            samples.append((counts, q))
        if not samples:
            raise ValueError(f"No earlier groups for {name} dispersion")
        counts = np.concatenate([s[0] for s in samples])
        q = np.concatenate([s[1] for s in samples])
        totals = np.array([s[0].sum() for s in samples])
        def objective(log_concentration):
            a = np.exp(log_concentration)
            return float(-np.sum(gammaln(a) - gammaln(totals + a)) - np.sum(gammaln(counts + a * q) - gammaln(a * q)))
        fit = minimize_scalar(objective, bounds=(np.log(.5), np.log(1000)), method="bounded")
        if not fit.success:
            raise ValueError("Allocation dispersion optimization failed")
        result[name] = {"concentration": float(np.exp(fit.x)), "calibration_team_games": len(samples)}
    return result


@dataclass
class TouchdownModel:
    """Historical offensive TD counts conditional on final team score."""
    bandwidth: float = 4.

    def fit(self, teams: pd.DataFrame) -> TouchdownModel:
        points = teams.points_for.to_numpy(dtype=int)
        touchdowns = (teams.passing_tds + teams.rushing_tds).to_numpy(dtype=int)
        capacities = points // 6
        if (touchdowns < 0).any() or (touchdowns > capacities).any():
            raise ValueError("Offensive touchdowns exceed the score budget")
        global_rate = float((touchdowns.sum() + .5) / (capacities.sum() + 1))
        self.pass_fraction_ = float((teams.passing_tds.sum() + .5) / (touchdowns.sum() + 1))
        self.probabilities_ = {}
        for score in range(101):
            support = np.arange(score // 6 + 1)
            weights = np.exp(-.5 * ((points - score) / self.bandwidth)**2)
            counts = np.array([weights[touchdowns == t].sum() for t in support])
            mass = counts + binom.pmf(support, score // 6, global_rate)
            self.probabilities_[score] = mass / mass.sum()
        self.fit_end_ = pd.to_datetime(teams.gameday).max()
        return self

    def sample(self, scores, *, rng) -> np.ndarray:
        scores = np.asarray(scores)
        if not np.isin(scores, list(self.probabilities_)).all():
            raise ValueError("Touchdown model supports integer scores 0–100")
        result = np.zeros(len(scores), dtype=int)
        for score in np.unique(scores):
            mask = scores == score
            probabilities = self.probabilities_[int(score)]
            result[mask] = rng.choice(len(probabilities), size=mask.sum(), p=probabilities)
        return result
