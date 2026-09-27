"""Point-estimate score models with explicit inputs and fold-local preprocessing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nfl_model.data_quality import require_columns
from nfl_model.modeling_dataset import feature_groups
from nfl_model.team_games import TARGET_COLUMNS

SCORE_TARGETS = ("home_score", "away_score")
CATEGORICAL_FEATURES = ("game_type", "surface")


def score_predictions(scores: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Nonnegative continuous scores, with algebraically consistent total/margin."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(index), 2) or not np.isfinite(scores).all():
        raise ValueError("Score predictions must be a finite (n_games, 2) array")
    scores = np.maximum(scores, 0.0)
    result = pd.DataFrame(scores, columns=SCORE_TARGETS, index=index)
    result["game_total"] = result.home_score + result.away_score
    result["home_margin"] = result.home_score - result.away_score
    return result


def _score_targets(games: pd.DataFrame) -> np.ndarray:
    require_columns(games, list(SCORE_TARGETS), "training games")
    scores = games[list(SCORE_TARGETS)].to_numpy(dtype=float, na_value=np.nan)
    if not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Training requires nonempty, finite, nonnegative scores")
    return scores


def _feature_frame(games: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    require_columns(games, list(columns), "model inputs")
    result = games.loc[:, list(columns)].copy()
    for col in columns:
        if col in CATEGORICAL_FEATURES:
            result[col] = result[col].astype(object).where(result[col].notna(), np.nan)
        else:
            result[col] = pd.to_numeric(result[col], errors="raise").astype(float)
            if np.isinf(result[col]).any():
                raise ValueError(f"Infinite values in model input {col}")
    return result


@dataclass
class RidgeScoreModel:
    """Ridge regression on home/away score means, not an outcome distribution.

    Alpha and the feature group are fixed before fit. Each fit creates a fresh
    pipeline. Predict reads only its saved allowlist, never target columns.
    """

    alpha: float = 100.0
    include_market: bool = False

    def fit(self, games: pd.DataFrame) -> RidgeScoreModel:
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        groups = feature_groups()
        self.feature_names_ = tuple(groups["football_features"] + (groups["market_features"] if self.include_market else []))
        X = _feature_frame(games, self.feature_names_)
        numeric = [col for col in self.feature_names_ if col not in CATEGORICAL_FEATURES]
        categorical = [col for col in self.feature_names_ if col in CATEGORICAL_FEATURES]
        preprocessing = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ], remainder="drop", sparse_threshold=0)
        self.pipeline_ = Pipeline([
            ("preprocess", preprocessing),
            ("regressor", Ridge(alpha=float(self.alpha), solver="lsqr", tol=1e-8)),
        ])
        self.pipeline_.fit(X, _score_targets(games))
        self.training_rows_ = len(games)
        self.training_seasons_ = sorted(int(s) for s in games.season.unique())
        return self

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "pipeline_"):
            raise ValueError("Fit the score model before prediction")
        X = _feature_frame(games, self.feature_names_)
        return score_predictions(self.pipeline_.predict(X), games.index)


@dataclass
class TrainingMeanModel:
    """Fixed home/away scoring means estimated only from the fitting seasons."""

    def fit(self, games: pd.DataFrame) -> TrainingMeanModel:
        self.means_ = _score_targets(games).mean(axis=0)
        return self

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "means_"):
            raise ValueError("Fit the mean benchmark before prediction")
        return score_predictions(np.tile(self.means_, (len(games), 1)), games.index)


@dataclass
class RecentTeamAverageModel(TrainingMeanModel):
    """Average of last-five team scoring and opponent points allowed.

    Each missing component falls back to the fitting sample's home/away scoring
    mean. No future outcome is used to fill an empty team history.
    """

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "means_"):
            raise ValueError("Fit the recent-team benchmark before prediction")
        scores = []
        for i, (side, other) in enumerate((("home", "away"), ("away", "home"))):
            columns = [f"{side}_pregame_points_for_last_5", f"{other}_pregame_points_against_last_5"]
            require_columns(games, columns, "recent-team benchmark")
            values = games[columns].astype(float).fillna(self.means_[i])
            scores.append(values.mean(axis=1).to_numpy())
        return score_predictions(np.column_stack(scores), games.index)


@dataclass
class MarketLineModel:
    """Direct spread/total benchmark. No odds timestamps or win probabilities."""

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        columns = ["market_total_line", "market_spread_line"]
        require_columns(games, columns, "market benchmark")
        total, margin = (pd.to_numeric(games[col], errors="raise").astype(float) for col in columns)
        if np.isinf(total).any() or np.isinf(margin).any():
            raise ValueError("Market lines must be finite or missing")
        if (total.dropna() < 0).any() or (margin.abs() > total).any():
            raise ValueError("Market lines imply negative team scores")
        # Preserve an available total or spread even if the other line is missing.
        return pd.DataFrame({
            "home_score": (total + margin) / 2,
            "away_score": (total - margin) / 2,
            "game_total": total,
            "home_margin": margin,
        }, index=games.index)[list(TARGET_COLUMNS)]
