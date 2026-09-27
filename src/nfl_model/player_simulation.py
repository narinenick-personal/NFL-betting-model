"""Joint score/volume draws and reconciled player opportunity/production draws."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from nfl_model.data_quality import require_unique_keys
from nfl_model.distributions import game_rng
from nfl_model.opportunity_models import VOLUME_TARGETS, OPPORTUNITIES
from nfl_model.player_dataset import OTHER


@dataclass
class ConditionalVolumeDistribution:
    def fit(self, score_errors: pd.DataFrame, volume_errors: pd.DataFrame, *, prediction_start):
        require_unique_keys(score_errors, ["game_id"], "score errors")
        require_unique_keys(volume_errors, ["game_id", "target"], "volume errors")
        for frame, cutoff in ((score_errors, "mean_fit_end"), (volume_errors, "fit_end")):
            dates = pd.to_datetime(frame.gameday)
            if not dates.lt(pd.Timestamp(prediction_start)).all() or not pd.to_datetime(frame[cutoff]).lt(dates).all():
                raise ValueError("Joint volume calibration must be earlier and out of time")
        residuals = volume_errors.assign(error=volume_errors.actual - volume_errors.predicted).pivot(index="game_id", columns="target", values="error")
        score = score_errors.set_index("game_id").sort_index()
        if set(score.index) != set(residuals.index) or len(score) < 3:
            raise ValueError("Joint score/volume residual cohorts must match")
        values = np.column_stack([score[["residual_home_score", "residual_away_score"]], residuals.loc[score.index, list(VOLUME_TARGETS)]])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite joint residuals")
        covariance = LedoitWolf().fit(values).covariance_ + np.eye(8) * 1e-8
        self.bias_ = values.mean(axis=0)
        self.gain_ = covariance[2:, :2] @ np.linalg.inv(covariance[:2, :2])
        self.covariance_ = covariance[2:, 2:] - self.gain_ @ covariance[:2, 2:]
        self.covariance_ = (self.covariance_ + self.covariance_.T) / 2 + np.eye(6) * 1e-8
        self.calibration_end_ = pd.to_datetime(score.gameday).max()
        self.calibration_ids_ = tuple(score.index)
        return self

    def sample(self, scores, score_means, volume_means, *, rng):
        scores, score_means, volume_means = (np.asarray(x, dtype=float) for x in (scores, score_means, volume_means))
        if scores.ndim != 2 or scores.shape[1] != 2 or score_means.shape != (2,) or volume_means.shape != (6,):
            raise ValueError("Need score draws, two score means and six volume means")
        if any(not np.isfinite(x).all() or (x < 0).any() for x in (scores, score_means, volume_means)):
            raise ValueError("Score and volume inputs must be finite and nonnegative")
        location = volume_means + self.bias_[2:] + (scores - score_means - self.bias_[:2]) @ self.gain_.T
        draws = np.maximum(location + rng.multivariate_normal(np.zeros(6), self.covariance_, size=len(scores)), 0)
        # Unbiased stochastic rounding for counts, not for NFL score outcomes.
        rounded = np.floor(draws).astype(int) + rng.binomial(1, draws % 1)
        return pd.DataFrame(rounded, columns=VOLUME_TARGETS)


def allocate_multinomial(totals, probabilities, *, rng):
    totals = np.asarray(totals, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape[0] != len(totals) or (totals < 0).any() or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid allocation totals/probabilities")
    sums = probabilities.sum(axis=1)
    if (sums <= 0).any():
        raise ValueError("Allocation probabilities need positive mass")
    probabilities = probabilities / sums[:, None]
    result = np.zeros(probabilities.shape, dtype=int)
    remaining, mass = totals.copy(), np.ones(len(totals))
    for i in range(probabilities.shape[1] - 1):
        p = np.divide(probabilities[:, i], mass, out=np.zeros(len(totals)), where=mass > 1e-15)
        result[:, i] = rng.binomial(remaining, np.clip(p, 0, 1))
        remaining -= result[:, i]
        mass -= probabilities[:, i]
    result[:, -1] = remaining
    return result


def allocate_with_capacities(totals, capacities, *, rng):
    """Uniform allocation without replacement; each unit has one available slot."""
    totals, capacities = np.asarray(totals, dtype=int), np.asarray(capacities, dtype=int)
    if (capacities < 0).any() or (totals < 0).any() or (totals > capacities.sum(axis=1)).any():
        raise ValueError("Allocation demand exceeds capacity")
    result = np.zeros_like(capacities)
    remaining, slots = totals.copy(), capacities.sum(axis=1)
    for i in range(capacities.shape[1] - 1):
        result[:, i] = rng.hypergeometric(capacities[:, i], slots - capacities[:, i], remaining)
        remaining -= result[:, i]
        slots -= capacities[:, i]
    result[:, -1] = remaining
    return result


def _random_share_counts(totals, weights, active, concentration, *, rng):
    raw = np.maximum(np.asarray(weights), 0)[None, :] * active
    no_mass = raw.sum(axis=1) == 0
    raw[no_mass, -1] = 1  # OTHER is always last and eligible.
    alpha = concentration * raw / raw.sum(axis=1)[:, None]
    gamma = rng.gamma(np.where(alpha > 0, alpha, 1.))
    gamma[alpha == 0] = 0
    no_mass = gamma.sum(axis=1) == 0
    gamma[no_mass, -1] = 1
    return allocate_multinomial(totals, gamma, rng=rng)


def simulate_team_players(candidates, weights, efficiencies, team_volume, scores, *, target_rate,
                          allocation_parameters, production_parameters, touchdown_model, rng):
    """Arrays are (draw, player); caller sorts named players then OTHER."""
    if candidates.iloc[-1].player_id != OTHER or candidates.is_other.sum() != 1:
        raise ValueError("Exactly one OTHER player must be the final candidate")
    draws, count = len(scores), len(candidates)
    active = rng.binomial(1, weights.active_probability.to_numpy(), size=(draws, count)).astype(bool)
    active[:, -1] = True
    target_totals = rng.binomial(team_volume["attempts"], target_rate)
    outputs = {"offense_active": active.astype(int)}
    for name in OPPORTUNITIES:
        totals = target_totals if name == "targets" else team_volume[name]
        outputs[name] = _random_share_counts(totals, weights[f"weight_{name}"], active,
                                             allocation_parameters[name]["concentration"], rng=rng)
    catch_p = np.broadcast_to(efficiencies.catch_probability.to_numpy(), (draws, count)).copy()
    rho = production_parameters["catch_intraclass_correlation"]
    if rho > 1e-8:
        concentration = 1 / rho - 1
        catch_p = rng.beta(catch_p * concentration, (1 - catch_p) * concentration)
    outputs["receptions"] = rng.binomial(outputs["targets"], catch_p)
    for name, exposure, forecast in (("receiving", "receptions", "receiving_yards_per_catch"), ("rushing", "carries", "rushing_yards_per_carry")):
        parameters = production_parameters[name]
        n = outputs[exposure]
        mean = efficiencies[forecast].to_numpy() + parameters["bias_per_opportunity"]
        shared_error = rng.normal(size=(draws, 1)) * np.sqrt(parameters["shared_variance"])
        independent_error = rng.normal(size=(draws, count)) * np.sqrt(n * parameters["individual_variance"])
        yards = np.rint(n * (mean + shared_error) + independent_error)
        outputs[f"{name}_yards"] = np.clip(yards, -99 * n, 99 * n).astype(int)
    total_receptions = outputs["receptions"].sum(axis=1)
    outputs["completions"] = allocate_with_capacities(total_receptions, outputs["attempts"], rng=rng)
    passing_yards = outputs["receiving_yards"].sum(axis=1)
    qb_weights = outputs["completions"].astype(float)
    qb_weights[qb_weights.sum(axis=1) == 0, -1] = 1
    outputs["passing_yards"] = allocate_multinomial(np.abs(passing_yards), qb_weights, rng=rng) * np.sign(passing_yards)[:, None]
    td_count = np.minimum(touchdown_model.sample(scores, rng=rng), total_receptions + team_volume["carries"])
    passing_td = rng.binomial(td_count, touchdown_model.pass_fraction_)
    passing_td = np.minimum(np.maximum(passing_td, td_count - team_volume["carries"]), total_receptions)
    rushing_td = td_count - passing_td
    outputs["receiving_tds"] = allocate_with_capacities(passing_td, outputs["receptions"], rng=rng)
    outputs["passing_tds"] = allocate_with_capacities(passing_td, outputs["completions"], rng=rng)
    outputs["rushing_tds"] = allocate_with_capacities(rushing_td, outputs["carries"], rng=rng)
    return outputs


@dataclass
class PlayerGameSimulator:
    score_model: object
    volume_model: object
    opportunity_model: object
    production_model: object
    volume_distribution: ConditionalVolumeDistribution
    target_rate: float
    allocation_parameters: dict
    production_parameters: dict
    touchdown_model: object
    fit_end: pd.Timestamp

    def simulate(self, games: pd.DataFrame, players: pd.DataFrame, *, draws=10_000, seed=20260925):
        """Yield one game's score, team-volume and player arrays; reads no targets."""
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
            raise ValueError("draws must be an integer >= 2")
        require_unique_keys(games, ["game_id"], "simulation games")
        require_unique_keys(players, ["game_id", "team", "player_id"], "simulation candidates")
        cutoff = max(pd.Timestamp(self.fit_end), self.volume_model.fit_end_, self.opportunity_model.fit_end_,
                     self.production_model.fit_end_, self.touchdown_model.fit_end_, pd.Timestamp(self.score_model.mean_fit_end),
                     self.score_model.distribution.score_history_end_, self.score_model.distribution.residual_model_.calibration_end_,
                     self.volume_distribution.calibration_end_, pd.Timestamp(self.production_parameters["calibration_end"]))
        if not pd.to_datetime(games.gameday).gt(cutoff).all():
            raise ValueError("Player simulations must follow all fitting/calibration cutoffs")
        means = self.score_model.mean_model.predict(games)
        volume = self.volume_model.predict(games)
        lookup = {key: group.sort_values(["is_other", "player_id"]).reset_index(drop=True) for key, group in players.groupby(["game_id", "team"], sort=False)}
        for row in games.itertuples():
            grid = self.score_model.distribution.predict_grid([means.loc[row.Index, "home_score"], means.loc[row.Index, "away_score"]], season=row.season, game_type=row.game_type)
            scores = grid.sample(draws=draws, rng=game_rng(seed, row.game_id, "player_score"))
            team_volume = self.volume_distribution.sample(scores[["home_score", "away_score"]], means.loc[row.Index, ["home_score", "away_score"]], volume.loc[row.Index], rng=game_rng(seed, row.game_id, "player_volume"))
            records = {}
            for side in ("home", "away"):
                team = getattr(row, f"{side}_team")
                if (row.game_id, team) not in lookup:
                    raise ValueError("Missing player candidate team")
                candidates = lookup[(row.game_id, team)]
                if candidates.loc[candidates.is_other.eq(0), "history_end"].isna().any() or candidates.history_end.dropna().ge(pd.Timestamp(row.gameday)).any():
                    raise ValueError("Candidate features must precede their game")
                weights = self.opportunity_model.predict_weights(candidates)
                efficiency = self.production_model.predict(candidates)
                tv = {name: team_volume[f"{side}_{name}"].to_numpy() for name in ("attempts", "carries", "sacks")}
                arrays = simulate_team_players(candidates, weights, efficiency, tv, scores[f"{side}_score"].to_numpy(),
                                               target_rate=self.target_rate, allocation_parameters=self.allocation_parameters,
                                               production_parameters=self.production_parameters, touchdown_model=self.touchdown_model,
                                               rng=game_rng(seed, row.game_id, f"players_{team}"))
                records[team] = {"candidates": candidates, "samples": arrays, "volume": tv}
            yield row.game_id, scores, team_volume, records
