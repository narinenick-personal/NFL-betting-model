"""A learned discrete score grid around trained, correlated final-score forecasts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from nfl_model.data_quality import require_columns, require_unique_keys
from nfl_model.distributions import JointResidualDistribution, game_rng
from nfl_model.settlement import SettlementRules, grade_market, overtime_format, validate_scores


@dataclass
class ScoreGrid:
    home: np.ndarray
    away: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self):
        home, away = validate_scores(self.home, self.away)
        weights = np.asarray(self.probabilities, dtype=float)
        if home.ndim != 1 or not len(home) or weights.shape != home.shape or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError("ScoreGrid requires aligned score vectors and normalized probabilities")
        self.home, self.away, self.probabilities = home.astype(int), away.astype(int), weights

    def market_probabilities(self, *, market: str, side: str, line: float | None = None, rules: SettlementRules = SettlementRules()) -> dict:
        grades = grade_market(self.home, self.away, market=market, side=side, line=line, rules=rules)
        return {"p_win": float(self.probabilities[grades == 1].sum()),
                "p_loss": float(self.probabilities[grades == -1].sum()),
                "p_push": float(self.probabilities[grades == 0].sum())}

    def sample(self, *, draws: int = 10_000, rng: np.random.Generator) -> pd.DataFrame:
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive integer")
        indices = rng.choice(len(self.probabilities), size=draws, p=self.probabilities)
        home, away = self.home[indices], self.away[indices]
        return pd.DataFrame({"home_score": home, "away_score": away, "game_total": home + away, "home_margin": home - away})


def _local_scoring_correction(counts: np.ndarray, bandwidth: float) -> np.ndarray:
    """Local frequency / smoothed frequency, with one pseudo-count and bounded lift.

    This retains local scoring spikes rather than multiplying by the complete
    unconditional score distribution, which would double-count its broad shape.
    """
    smooth = gaussian_filter1d(counts.astype(float), bandwidth, mode="constant")
    return np.clip((counts + 1) / (smooth + 1), .25, 4.0)


@dataclass
class DiscreteScoreDistribution:
    max_score: int = 100
    bandwidth: float = 2.0

    def fit(self, residuals: pd.DataFrame, history: pd.DataFrame, *, prediction_start) -> DiscreteScoreDistribution:
        if not isinstance(self.max_score, int) or self.max_score < 10 or not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("Invalid discrete score-grid configuration")
        require_unique_keys(history, ["game_id"], "score history")
        require_columns(history, ["gameday", "season", "game_type", "home_score", "away_score"], "score history")
        dates = pd.to_datetime(history.gameday)
        if dates.isna().any() or not dates.lt(pd.Timestamp(prediction_start)).all():
            raise ValueError("Score history must precede the prediction period")
        home, away = validate_scores(history.home_score, history.away_score)
        if (home > self.max_score).any() or (away > self.max_score).any() or (home == 1).any() or (away == 1).any():
            raise ValueError("History includes scores outside this model's common-scoring support")
        for row in history.itertuples(index=False):
            rule = overtime_format(row.season, row.game_type)
            if not rule["final_tie_allowed"] and row.home_score == row.away_score:
                raise ValueError("A postseason final cannot be tied")
        self.residual_model_ = JointResidualDistribution("gaussian").fit(residuals, prediction_start=prediction_start)
        require_columns(residuals, ["pred_home_score", "pred_away_score"], "residual predictions")
        linked = residuals.merge(history[["game_id", "gameday", "home_score", "away_score", "game_type"]], on="game_id", suffixes=("", "_history"), validate="one_to_one")
        if len(linked) != len(residuals):
            raise ValueError("Every calibration residual must match a historical game")
        if not pd.to_datetime(linked.gameday).eq(pd.to_datetime(linked.gameday_history)).all():
            raise ValueError("Residual dates disagree with historical games")
        for side in ("home", "away"):
            target = f"{side}_score_history" if f"{side}_score_history" in linked else f"{side}_score"
            if not np.allclose(linked[f"pred_{side}_score"] + linked[f"residual_{side}_score"], linked[target]):
                raise ValueError("Residuals disagree with historical scores")
        count = np.bincount(np.concatenate([home, away]).astype(int), minlength=self.max_score + 1)
        margins = (home - away).astype(int)
        # Symmetry leaves home advantage in the mean forecasts, not in this prior.
        margin_counts = np.bincount(np.concatenate([margins, -margins]) + self.max_score, minlength=2 * self.max_score + 1)
        self.score_correction_ = _local_scoring_correction(count, self.bandwidth)
        self.margin_correction_ = _local_scoring_correction(margin_counts, self.bandwidth)
        self.margin_correction_[self.max_score] = 1.0  # Tie odds calibrated separately.
        support = np.array([s for s in range(self.max_score + 1) if s != 1], dtype=int)
        h, a = np.meshgrid(support, support, indexing="ij")
        self.home_, self.away_ = h.ravel(), a.ravel()
        self.coordinates_ = np.column_stack([self.home_, self.away_])
        self.diagonal_ = self.home_ == self.away_
        self.log_correction_ = (np.log(self.score_correction_[self.home_]) + np.log(self.score_correction_[self.away_])
                                + np.log(self.margin_correction_[self.home_ - self.away_ + self.max_score]))
        self.precision_ = np.linalg.inv(self.residual_model_.covariance_)
        self.score_history_end_ = dates.max()
        self.score_history_ids_ = tuple(history.game_id)
        regular = history.loc[history.game_type.eq("REG")]
        if regular.empty:
            raise ValueError("Regular-season history is needed to estimate final ties")
        self.tie_prior_rate_ = float((regular.home_score.eq(regular.away_score).sum() + .5) / (len(regular) + 1))
        linked_type = "game_type_history" if "game_type_history" in linked else "game_type"
        calibration = linked.loc[linked[linked_type].eq("REG")]
        if calibration.empty:
            raise ValueError("Need regular-season out-of-time predictions for tie calibration")
        tie_probabilities = np.array([self._base_pmf([row.pred_home_score, row.pred_away_score])[self.diagonal_].sum() for row in calibration.itertuples(index=False)])
        def objective(log_scale):
            odds = np.exp(log_scale)
            return np.mean(odds * tie_probabilities / (1 + (odds - 1) * tie_probabilities)) - self.tie_prior_rate_
        self.tie_scale_ = float(np.exp(brentq(objective, -30, 30)))
        self.regular_history_games_ = len(regular)
        self.regular_history_ties_ = int(regular.home_score.eq(regular.away_score).sum())
        self.regime_counts_ = {}
        for row in history.itertuples(index=False):
            name = overtime_format(row.season, row.game_type)["regime"]
            self.regime_counts_[name] = self.regime_counts_.get(name, 0) + 1
        return self

    def _base_pmf(self, score_mean) -> np.ndarray:
        mean = np.asarray(score_mean, dtype=float)
        if mean.shape != (2,) or not np.isfinite(mean).all() or (mean < 0).any():
            raise ValueError("Need two finite nonnegative projected scores")
        location = mean + self.residual_model_.bias_
        std = np.sqrt(np.diag(self.residual_model_.covariance_))
        # Guard against silent truncation of meaningful upper-tail probability.
        if norm.sf((self.max_score + .5 - location) / std).sum() > 1e-6:
            raise ValueError("Score grid is too small for this forecast; increase max_score")
        difference = self.coordinates_ - location
        log_mass = -.5 * np.einsum("ni,ij,nj->n", difference, self.precision_, difference) + self.log_correction_
        return np.exp(log_mass - logsumexp(log_mass))

    def predict_grid(self, score_mean, *, season: int, game_type: str) -> ScoreGrid:
        if not hasattr(self, "tie_scale_"):
            raise ValueError("Fit the discrete score distribution first")
        rules = overtime_format(season, game_type)
        weights = self._base_pmf(score_mean)
        weights[self.diagonal_] *= self.tie_scale_ if rules["final_tie_allowed"] else 0
        if weights.sum() <= 0:
            raise ValueError("No supported terminal scores remain; residual distribution is degenerate")
        weights /= weights.sum()
        return ScoreGrid(self.home_.copy(), self.away_.copy(), weights)


@dataclass
class DiscreteGameModel:
    mean_model: object
    distribution: DiscreteScoreDistribution
    mean_fit_end: pd.Timestamp

    def grids(self, games: pd.DataFrame):
        require_unique_keys(games, ["game_id"], "forecast games")
        require_columns(games, ["gameday", "season", "game_type"], "forecast games")
        dates = pd.to_datetime(games.gameday)
        cutoff = max(pd.Timestamp(self.mean_fit_end), self.distribution.score_history_end_, self.distribution.residual_model_.calibration_end_)
        if dates.isna().any() or not dates.gt(cutoff).all():
            raise ValueError("Forecast games must follow all fitting cutoffs")
        means = self.mean_model.predict(games)
        for row, mean in zip(games.itertuples(index=False), means.itertuples(index=False)):
            yield row.game_id, self.distribution.predict_grid([mean.home_score, mean.away_score], season=row.season, game_type=row.game_type)

    def simulate(self, games: pd.DataFrame, *, draws: int = 10_000, seed: int = 20260925):
        for game_id, grid in self.grids(games):
            yield game_id, grid.sample(draws=draws, rng=game_rng(seed, game_id, "discrete_scores"))
