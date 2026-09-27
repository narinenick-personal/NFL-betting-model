"""Train-only team volume, offensive participation, and player share estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nfl_model.data_quality import require_columns, require_unique_keys
from nfl_model.modeling_dataset import feature_groups
from nfl_model.models import _feature_frame, CATEGORICAL_FEATURES
from nfl_model.player_dataset import PLAYER_FEATURES

VOLUME_COMPONENTS = ("attempts", "carries", "sacks")
VOLUME_TARGETS = tuple(f"{side}_{name}" for side in ("home", "away") for name in VOLUME_COMPONENTS)
OPPORTUNITIES = ("attempts", "carries", "targets")
SHARE_NAMES = {"attempts": "attempt_share", "carries": "carry_share", "targets": "target_share"}


def preprocessing(columns, categorical) -> ColumnTransformer:
    numeric = [c for c in columns if c not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                               ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), list(categorical)),
    ], sparse_threshold=0)


def player_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    require_columns(rows, list(PLAYER_FEATURES), "player predictors")
    frame = rows[list(PLAYER_FEATURES)].copy()
    frame["position"] = frame.position.astype(object).where(frame.position.notna(), np.nan)
    for column in PLAYER_FEATURES:
        if column != "position":
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if np.isinf(frame.drop(columns="position").to_numpy()).any():
        raise ValueError("Infinite player features")
    return frame


def build_volume_games(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    require_unique_keys(games, ["game_id"], "volume games")
    require_unique_keys(teams, ["game_id", "team"], "volume team rows")
    result = games.copy()
    for side in ("home", "away"):
        selected = teams[["game_id", "team", "pass_attempts", "rush_attempts", "sacks_allowed"]].rename(
            columns={"team": f"{side}_team", "pass_attempts": f"{side}_attempts", "rush_attempts": f"{side}_carries", "sacks_allowed": f"{side}_sacks"})
        result = result.merge(selected, on=["game_id", f"{side}_team"], how="left", validate="one_to_one")
    values = result[list(VOLUME_TARGETS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Missing or invalid team volume outcomes")
    return result


@dataclass
class TeamVolumeModel:
    kind: str = "ridge"
    alpha: float = 1000.

    def fit(self, games: pd.DataFrame) -> TeamVolumeModel:
        self.columns_ = tuple(feature_groups()["football_features"])
        values = games[list(VOLUME_TARGETS)].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Training volume targets must be finite and nonnegative")
        self.means_ = values.mean(axis=0)
        self.fit_end_ = pd.to_datetime(games.gameday).max()
        if self.kind == "ridge":
            if not np.isfinite(self.alpha) or self.alpha <= 0:
                raise ValueError("alpha must be positive")
            self.pipeline_ = Pipeline([("preprocess", preprocessing(self.columns_, CATEGORICAL_FEATURES)),
                                       ("regressor", Ridge(alpha=self.alpha, solver="lsqr", tol=1e-8))])
            self.pipeline_.fit(_feature_frame(games, self.columns_), values)
        elif self.kind not in ("training_mean", "recent_average"):
            raise ValueError("Unknown volume model")
        return self

    def predict(self, games: pd.DataFrame) -> pd.DataFrame:
        if self.kind == "ridge":
            values = self.pipeline_.predict(_feature_frame(games, self.columns_))
        else:
            values = np.tile(self.means_, (len(games), 1))
            if self.kind == "recent_average":
                for i, (side, metric) in enumerate((s, m) for s in ("home", "away") for m in ("pass_attempts", "rush_attempts", "sacks_allowed")):
                    observed = games[f"{side}_pregame_{metric}_last_5"].to_numpy(dtype=float)
                    values[:, i] = np.where(np.isfinite(observed), observed, values[:, i])
        if not np.isfinite(values).all():
            raise ValueError("Volume model emitted nonfinite means")
        return pd.DataFrame(np.maximum(values, 0), index=games.index, columns=VOLUME_TARGETS)


def normalized_shares(rows: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Expected-active-weight approximation for point forecasts; no targets read."""
    require_unique_keys(rows, ["game_id", "team", "player_id"], "allocation candidates")
    if not rows.groupby(["game_id", "team"]).is_other.sum().eq(1).all():
        raise ValueError("Every team needs exactly one OTHER allocation row")
    result = pd.DataFrame(index=rows.index)
    for name in OPPORTUNITIES:
        raw = weights[f"weight_{name}"].to_numpy() * weights.active_probability.to_numpy()
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("Invalid opportunity allocation weights")
        series = pd.Series(raw, index=rows.index)
        sums = series.groupby([rows.game_id, rows.team]).transform("sum")
        result[name] = np.where(sums > 0, series / sums.where(sums > 0), rows.is_other.astype(float))
    return result


@dataclass
class PlayerOpportunityModel:
    """Participation hurdle and conditional share rates, with a recent-use baseline."""
    kind: str = "trained"
    alpha: float = .05

    def fit(self, rows: pd.DataFrame) -> PlayerOpportunityModel:
        require_columns(rows, ["offense_active", *OPPORTUNITIES, *(f"team_{c}" for c in OPPORTUNITIES)], "opportunity outcomes")
        self.fit_end_ = pd.to_datetime(rows.gameday).max()
        self.position_activity_ = rows.loc[rows.is_other.eq(0)].groupby("position").offense_active.mean().to_dict()
        self.global_activity_ = float(rows.loc[rows.is_other.eq(0), "offense_active"].mean())
        self.prior_rates_ = {}
        for name in OPPORTUNITIES:
            grouped = rows.groupby("position")[[name, f"team_{name}"]].sum()
            self.prior_rates_[name] = (grouped[name] / grouped[f"team_{name}"].clip(lower=1)).to_dict()
        if self.kind == "recent_average":
            return self
        if self.kind != "trained":
            raise ValueError("Unknown player opportunity model")
        self.preprocessor_ = preprocessing(PLAYER_FEATURES, ("position",))
        X = self.preprocessor_.fit_transform(player_inputs(rows))
        known = rows.offense_active.notna() & rows.is_other.eq(0)
        y = rows.loc[known, "offense_active"].to_numpy(dtype=float)
        if not len(y) or not np.isin(y, [0, 1]).all():
            raise ValueError("Known binary offensive participation is required")
        self.activity_constant_ = float((y.sum() + .5) / (len(y) + 1))
        self.activity_ = None
        if len(np.unique(y)) == 2:
            self.activity_ = LogisticRegression(C=1., max_iter=500).fit(X[known], y)
        self.rates_ = {}
        for name in OPPORTUNITIES:
            include = (rows.offense_active.eq(1) | rows.is_other.eq(1)) & rows[f"team_{name}"].gt(0)
            exposure = rows.loc[include, f"team_{name}"].to_numpy(dtype=float)
            response = rows.loc[include, name].to_numpy(dtype=float) / exposure
            if not len(response) or response.sum() <= 0:
                raise ValueError(f"No training opportunity for {name}")
            self.rates_[name] = PoissonRegressor(alpha=self.alpha, max_iter=500, tol=1e-7).fit(X[include], response, sample_weight=exposure)
        return self

    def predict_weights(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.kind == "trained":
            X = self.preprocessor_.transform(player_inputs(rows))
            active = self.activity_.predict_proba(X)[:, 1] if self.activity_ is not None else np.full(len(rows), self.activity_constant_)
            rates = {name: self.rates_[name].predict(X) for name in OPPORTUNITIES}
        else:
            active = rows.prior_offense_active_last5.fillna(rows.position.map(self.position_activity_)).fillna(self.global_activity_).to_numpy()
            rates = {}
            for name in OPPORTUNITIES:
                prior = rows[f"prior_{SHARE_NAMES[name]}_last5"].fillna(rows.position.map(self.prior_rates_[name])).fillna(0).to_numpy()
                # Rolling shares are unconditional; undo the hurdle for the common API.
                rates[name] = prior / np.maximum(active, .01)
        active = np.where(rows.is_other.eq(1), 1., np.clip(active, 0, 1))
        result = pd.DataFrame({"active_probability": active}, index=rows.index)
        for name in OPPORTUNITIES:
            result[f"weight_{name}"] = np.maximum(rates[name], 0)
        return result

    def predict_shares(self, rows: pd.DataFrame) -> pd.DataFrame:
        return normalized_shares(rows, self.predict_weights(rows))
