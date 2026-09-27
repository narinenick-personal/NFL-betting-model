"""Joint score uncertainty learned exclusively from earlier out-of-time errors."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from nfl_model.data_quality import require_columns, require_unique_keys
from nfl_model.team_games import TARGET_COLUMNS

DISTRIBUTIONS = ("paired_bootstrap", "gaussian")
RESIDUAL_COLUMNS = ("residual_home_score", "residual_away_score")


def game_rng(seed: int, game_id: str, stream: str) -> np.random.Generator:
    """Stable across process restarts, row order, and batches; no Python hash()."""
    digest = hashlib.sha256(f"{seed}|{game_id}|{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


@dataclass
class JointResidualDistribution:
    kind: str = "paired_bootstrap"

    def fit(self, residuals: pd.DataFrame, *, prediction_start: pd.Timestamp) -> JointResidualDistribution:
        """Each residual must have a mean-model fitting cutoff before its game.

        The calibration games must all precede the first forecast date. Raw error
        means are retained as an estimated bias correction, never centered away.
        """
        if self.kind not in DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution: {self.kind}")
        require_columns(residuals, ["game_id", "gameday", "mean_fit_end", *RESIDUAL_COLUMNS], "calibration residuals")
        require_unique_keys(residuals, ["game_id"], "calibration residuals")
        rows = residuals.copy()
        rows["gameday"] = pd.to_datetime(rows.gameday)
        rows["mean_fit_end"] = pd.to_datetime(rows.mean_fit_end)
        prediction_start = pd.Timestamp(prediction_start)
        if rows[["gameday", "mean_fit_end"]].isna().any().any() or pd.isna(prediction_start):
            raise ValueError("Calibration dates must not be missing")
        if not rows.mean_fit_end.lt(rows.gameday).all():
            raise ValueError("Residuals must come from out-of-time mean predictions")
        if not rows.gameday.lt(prediction_start).all():
            raise ValueError("Calibration residuals overlap the forecast period")
        rows = rows.sort_values(["gameday", "game_id"])
        values = rows[list(RESIDUAL_COLUMNS)].to_numpy(dtype=float)
        if len(values) < 3 or not np.isfinite(values).all():
            raise ValueError("At least three finite paired residuals are required")
        self.residuals_ = values
        self.bias_ = values.mean(axis=0)
        # Estimated shrinkage regularizes the two-score covariance; no future tuning.
        covariance = LedoitWolf().fit(values)
        self.covariance_ = covariance.covariance_ + np.eye(2) * 1e-8
        self.shrinkage_ = float(covariance.shrinkage_)
        self.calibration_rows_ = len(values)
        self.calibration_end_ = rows.gameday.max()
        self.calibration_game_ids_ = tuple(rows.game_id)
        return self

    def sample(self, score_mean, *, draws: int = 10_000, rng: np.random.Generator) -> np.ndarray:
        """Return (draws, 4) continuous scores, total and margin.

        Lower-censor scores at zero; this changes moments and adds mass at zero.
        This is not a discrete NFL scoring/overtime model and must not price pushes.
        """
        if not hasattr(self, "residuals_"):
            raise ValueError("Fit the residual distribution first")
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
            raise ValueError("draws must be an integer of at least two")
        mean = np.asarray(score_mean, dtype=float)
        if mean.shape != (2,) or not np.isfinite(mean).all() or (mean < 0).any():
            raise ValueError("Expected two finite nonnegative score means")
        if self.kind == "paired_bootstrap":
            # One index per game outcome preserves within-game dependence.
            errors = self.residuals_[rng.integers(0, len(self.residuals_), size=draws)]
        else:
            errors = rng.multivariate_normal(self.bias_, self.covariance_, size=draws)
        scores = np.maximum(mean + errors, 0.0)
        return np.column_stack([scores, scores.sum(axis=1), scores[:, 0] - scores[:, 1]])


@dataclass
class JointScoreModel:
    mean_model: object
    distribution: JointResidualDistribution
    mean_fit_end: pd.Timestamp

    def simulate(self, games: pd.DataFrame, *, draws: int = 10_000, seed: int = 20260925):
        """Yield one game's draws at a time, without reading outcome columns."""
        require_unique_keys(games, ["game_id"], "simulation games")
        require_columns(games, ["gameday"], "simulation games")
        dates = pd.to_datetime(games.gameday)
        cutoff = max(pd.Timestamp(self.mean_fit_end), self.distribution.calibration_end_)
        if dates.isna().any() or not dates.gt(cutoff).all():
            raise ValueError("Simulation dates must follow mean-model and calibration cutoffs")
        means = self.mean_model.predict(games)[["home_score", "away_score"]].to_numpy(dtype=float)
        for game_id, mean in zip(games.game_id, means):
            samples = self.distribution.sample(mean, draws=draws, rng=game_rng(seed, game_id, self.distribution.kind))
            yield game_id, pd.DataFrame(samples, columns=TARGET_COLUMNS)
