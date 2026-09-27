# NFL Betting Model

The project now includes team and player datasets, chronological model validation,
correlated game/player simulations, game-market and player-prop pricing, archived
quote replay, a live odds workspace, and a local Streamlit dashboard. Model
forecasts are a **2025 historical research replay**, fitted through the 2024
season. The separate live odds board accepts manual quotes, CSV imports and an
optional API feed; current-game forecasts and injury/active-list feeds are not
connected yet.

Launch the completed local dashboard:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Open `http://127.0.0.1:8501`. Views cover game markets, player props, live odds, archived
quote replay, model validation, and data coverage. The dashboard loads saved
models; opening it does not retrain them. The verified environment is recorded
in `requirements.lock.txt`; `pip install -r requirements.lock.txt` reproduces
those dependency versions.

## Live odds and lines

Choose **Live odds & lines** in the sidebar. This view works even without the
historical datasets or saved models. It supports:

- Manual sportsbook quotes for moneylines, spreads, totals and player props.
- CSV import/export with the empty template `data/templates/live_odds.csv`.
- The Odds API for NFL game markets and selected-event player props.
- Latest observed main lines by book/selection, quote age, stale/started flags,
  quoted break-even probabilities and highest payouts at identical lines.
- An optional calculator using **your own** conditional win and push/void
  estimates. It does not present historical models as current-game forecasts.

Spread `line` is the **selected team's handicap** in this board: away +3.5 is
entered as +3.5. This differs from the historical game grid's home-handicap API.
Moneyline lines and nonapplicable player fields must be empty. CSV timestamps
need an explicit timezone; `quoted_at` must not be future-dated and must precede
kickoff. Manual saves timestamp the quote at save time. Whole and half lines are
supported. This is a pregame workspace, not an in-play model.

To connect [The Odds API](https://the-odds-api.com/liveapi/guides/v4/), use a
provider account and either enter a key in the password field, set the
`ODDS_API_KEY` environment variable, or copy `.streamlit/secrets.toml.example`
to `.streamlit/secrets.toml` and fill in the key. For a hosted deployment, add
the following in [Streamlit Cloud's Secrets settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management):

```toml
ODDS_API_KEY = "your-key-here"
```

The real secrets file is ignored by Git. No key is included in quote exports,
model artifacts, or user-facing transport errors. Requests use the provider's
fixed HTTPS host, a timeout, no automatic retries, and explicit refresh buttons.
Game markets are selected for one region; props are fetched one event at a time.
Provider quota headers and last-fetch time are displayed. A server key shares
its provider quota across app visitors; for a public app, users can enter their
own key instead. No account signup or purchase is performed by this application.

The board retains provider update times, not refresh times, for quote age.
Repeated identical imports are idempotent; conflicting quote IDs are rejected.
It shows only the newest main line per book/selection, with previous snapshots
in the history export. It is not an alternate-line catalog. Highest-payout
comparisons require matching events, sides, lines, statistics and stated
participation contracts. API prop settlement rules are marked `book_rules`
(unverified), excluded from cross-book best-price flags, and not mapped to
historical player IDs. Listed availability and player names remain provider
claims; a recent quote may have moved or been withdrawn.

Quotes are held **only in each browser session**, capped at 20,000 observations.
Download the history before reloading or leaving and reimport it later. This
avoids writing shared visitor data to the repository or relying on ephemeral
cloud storage. There is no durable database or automatic background collection.
The snapshot age flags update on app interaction, including **Recheck quote
ages**. API refreshes preserve saved history if the provider fails. Credentialed
feed access must be verified with your own account; automated tests use mocked
provider responses, not real sportsbook prices.

## Setup and build

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_modeling_dataset.py --check-tests
```

The build defaults to NFL seasons **2021 through 2025**, including completed
regular-season and playoff games. Season means the NFL season, so its January
and February games stay in that season's history. Downloads use nflreadpy and
are cached in `data/raw/`, anchored to this repository regardless of the shell's
working directory. Repeated builds use those snapshots. `--refresh` requests
fresh data through nflreadpy; its own in-process cache may still apply.

```bash
python scripts/build_modeling_dataset.py --seasons 2021 2022 2023 2024 2025
python -m pytest -q
```

The build writes:

| File under `data/processed/` | Purpose |
| --- | --- |
| `game_modeling_dataset.parquet` | One row per completed game: identifiers, pregame predictors, markets, recorded context, targets |
| `team_game_dataset.parquet` | Two rows per game: schedule identity, raw observations, derived metrics, and shifted history |
| `game_modeling_manifest.json` | Explicit column roles, metric formulas, history policy, dependency versions, source file hashes |
| `game_modeling_report.json` | Coverage, duplicate counts, missing percentages for every column and raw metric, test result |
| `game_modeling_report.txt` | Concise human-readable build report and first five games |

`--output-dir PATH` changes the output location. `--check-tests` runs the offline
test suite first and stops on failure. The build only reports tests as passed
when it actually ran them.

## Using the dataset safely

Do **not** train on every numeric column or on the team audit table. Load the
explicit feature contract, or use the supplied selector with `PYTHONPATH=src`:

```python
import pandas as pd
from nfl_model.modeling_dataset import select_model_features

# Run from the repository root.
games = pd.read_parquet("data/processed/game_modeling_dataset.parquet")
X_football = select_model_features(games)
X_with_market = select_model_features(games, include_market=True)
y = games[["home_score", "away_score", "game_total", "home_margin"]]
```

The manifest records the same roles for programs that do not import the package.
Football features contain the home and away teams' shifted performance summaries,
prior-game counts, week, game type, designated home/away flags, rest, neutral-site
indicator, division matchup, and surface. These include categorical columns;
fit encoding and missing-value handling on the training split only.

All sportsbook fields have a `market_` prefix. They are excluded from the
football feature list. `market_spread_line` keeps nflverse's convention: positive
means the home team is favored; home ATS margin would therefore be
`home_margin - market_spread_line`. These snapshots contain no reliable price
observation timestamp, so they cannot establish an earlier executable price or
closing-line value. See the [nflverse schedule dictionary](https://nflreadr.nflverse.com/articles/dictionary_schedules.html).

Recorded `temp`, `wind`, and actual `roof` status are retained as a separate
`observed_context` group and excluded from default predictors. To use them in a
backtest at a specific pregame decision time, first obtain the forecasts or roof
announcements available then. No current-game QB identity, overtime, current-game
box score, or result/total alias enters the predictor list.

Targets are recomputed from schedule scores:

- `home_score` and `away_score`: designated teams' final scores.
- `game_total`: home score plus away score.
- `home_margin`: home score minus away score.

## Team metrics and historical windows

There are 36 base metrics. Raw game measurements live **only** in the audit
table; predictors use previous 3 games, previous 5 games, and season-to-date means.
Each operation groups by `(team, season)`, orders by actual game date, and shifts
by one **before** rolling or expanding. Postseason can use that season's regular
season. A new NFL season starts with no history. Bye weeks do not consume a game
window. A team's first game has missing performance summaries and zero prior-game
counts. Missing values are never backfilled using later games or other seasons.

| Metric | Definition |
| --- | --- |
| Points for / against | Team and opponent schedule scores, including all scoring phases |
| Pass attempts / rush attempts | `attempts` / `carries` |
| Pass dropbacks | Attempts + sacks suffered; excludes scrambles recorded as rushes |
| Total offensive plays | Attempts + carries + sacks suffered; excludes penalty-only/no-play snaps and two-point tries |
| Passing yards per attempt / rushing yards per carry | Respective yards divided by attempts or carries |
| Passing EPA per attempt / per dropback | Source passing EPA divided by attempts, or by attempts + sacks |
| Rushing EPA per carry | Source rushing EPA / carries |
| CPOE | Source `passing_cpoe`, preserving its percentage-point scale |
| Sack rate / interception rate | Sacks / pass dropbacks; interceptions thrown / pass attempts |
| Offensive fumbles lost | Sack + rushing + receiving fumbles lost; excludes return-game losses |
| Fumble lost rate / turnover rate | Offensive fumbles lost / plays; (interceptions + offensive fumbles lost) / plays |
| Team fumbles lost | All-phase `fumbles_lost_total`, retained separately |
| Offensive first-down rate | (Passing + rushing first downs) / plays; excludes penalty first downs |
| Explosive passing / rushing rate | `passing_20` / attempts; `rushing_10` / carries (20+ passing yards, 10+ rushing yards) |
| Penalties / penalty yards per game | Team counts supplied by nflverse, including all phases |
| Offensive scoring efficiency | (Passing + rushing touchdowns) / plays; a TD-per-play measure, not red-zone or drive efficiency |
| Defensive production | Sacks, interceptions and QB hits per game |

Passing/rushing yards, EPA, sacks allowed, interceptions, offensive first downs,
offensive touchdowns, and turnovers are also retained as game-level volumes and
lagged averages. The complete 36-metric formula map is saved in the manifest.

The supplied passing EPA includes sacks, so the added per-dropback metric is
useful alongside the requested per-attempt normalization. The source fields are
documented in the [nflverse stats dictionaries](https://nflreadr.nflverse.com/articles/dictionary_team_stats.html)
and [passing EPA documentation](https://nflreadr.nflverse.com/articles/dictionary_player_stats.html).
Rates with zero or missing denominators are missing, not infinite or zero.
Unavailable optional statistics remain missing. Rolling rate summaries are
**arithmetic means of game-level rates**, not pooled numerator/denominator ratios.
Within a fixed game window, missing observations are skipped (`min_periods=1`).
Prior-game counts describe games played, not each metric's nonmissing count.

The code checks duplicate and null keys, correct opponents/season/week/season
type, exactly two team records per completed game, valid scores, missing completed
game stats, and unknown schedule IDs. It rejects ambiguous same-team/same-day
history. Unplayed games with both scores missing and preseason games are excluded;
a partially missing score is an error. No unmatched completed records are silently
dropped. Pass inputs covering the same seasons to the public builder.

These are current nflverse snapshots, not historical as-of archives. Shifting
prevents game-outcome leakage through the feature calculations, but cannot undo
later upstream stat corrections or retrospective changes to EPA/CPOE models.
Strict point-in-time evaluation needs archived source vintages. Source parquet
hashes preserve which snapshots this build used.

## Modules and validation

- `data_loader.py`: cached NFL data access and explicit refresh for schedules/team stats.
- `data_quality.py`: required columns and key checks.
- `team_games.py`: schedule normalization and validated team/stat joins.
- `team_metrics.py`: named raw/derived game observations and formulas.
- `features.py`: shifted rolling and season-to-date history; legacy functions remain available.
- `modeling_dataset.py`: home/away assembly, feature roles and safe model-input selector.
- `reporting.py`: dataset manifest, missingness and build summaries.
- `scripts/build_modeling_dataset.py`: reproducible build entry point.
- `tests/`: offline synthetic-data tests, including outcome perturbations, shuffled
  input/prefix invariance, season resets, joins, rates, missing data, caching and
  artifact round trips.

## Train and evaluate baseline models

After building the dataset:

```bash
python scripts/train_baselines.py --check-tests
```

The default experiment is fixed before looking at its held-out results:

1. Fit candidates on **2021–2023** (854 games).
2. Choose the regularization strength using **2024** (285 games), minimizing
   pooled home/away score RMSE. The fixed alpha grid is 0.1, 1, 10, 100, 1,000,
   and 10,000; exact ties favor stronger regularization.
3. Refit the chosen model on **2021–2024** (1,139 games).
4. Evaluate once on **2025** (285 games, including that NFL season's playoffs).

There are five comparators:

| Model | Inputs and behavior |
| --- | --- |
| `training_mean` | Separate home/away scoring means from the fitting seasons |
| `recent_team_average` | Average of team points scored and opponent points allowed over their previous five games; missing components use the fitting sample's home/away mean |
| `market_lines` | Spread and total as margin/total forecasts, with team scores `(total ± spread) / 2` |
| `ridge_football` | Ridge score regression using the 231 football predictors |
| `ridge_market` | The same regression with the eight market inputs added |

Ridge predicts home/away scores together with a shared regularization strength.
Negative score means are clipped at zero; predictions remain continuous.
`game_total` and `home_margin` are derived from the two scores, so the four outputs
always reconcile. These are point estimates, not simulated scores or betting
probabilities. The market benchmark preserves raw available lines and does not
impute missing lines.

Each candidate fits its own pipeline: numeric median imputation with missing
indicators, standard scaling, categorical missing-value encoding and one-hot
encoding. All-missing training columns are retained with a zero fill. Unknown
future categories are ignored. Neither validation nor test features influence
fitted imputation/scaling/encoding parameters. See the scikit-learn
[pipeline leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
and [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

The model coefficients stay fixed during each evaluation season. Its pregame
rolling inputs advance using already completed games, as they would during the
season. This is evaluation of pregame predictions, not a preseason forecast of
the entire schedule. Date and season checks reject overlapping splits; the
training API does not use a random train/test split.

The command writes the following to `artifacts/baselines/`:

| Artifact | Contents |
| --- | --- |
| `report.md` | Validation/test RMSE and test MAE/bias, coverage and limitations |
| `predictions.parquet` | Every validation/test prediction, actual target, residual, model identity and fitting cutoff |
| `metrics.parquet` | MAE, RMSE, bias and sample coverage by model/split/target, on available and common-game cohorts |
| `tuning.parquet` | Every candidate's validation errors and selected alpha |
| `models.joblib` | Final model pipelines and benchmark objects fitted through 2024 |
| `run.json` | Split protocol, selected settings, feature lists, versions, source/dataset/model hashes and test result |

The command reloads its newly saved model bundle and checks that it reproduces
all test predictions. Residuals are **actual minus predicted**; reported bias is
**predicted minus actual**. Common-game comparisons use a target-specific
intersection across all models. If no common games exist, metrics are missing
with zero coverage, not reported as zero error.

To use the locally generated model bundle with `PYTHONPATH=src`:

```python
import joblib
import pandas as pd

models = joblib.load("artifacts/baselines/models.joblib")
games = pd.read_parquet("data/processed/game_modeling_dataset.parquet")
predictions = models["ridge_football"].predict(games.loc[games.season == 2025])
```

Only load trusted local joblib artifacts. Prediction needs the saved feature
allowlist; target columns and observed weather are not required. The builder
currently produces completed historical games, so a future-fixture feature
builder is still needed before live forecasting.

Optional CLI arguments include `--dataset`, `--output-dir`, `--train-seasons`,
`--validation-season`, `--test-season`, and `--alphas`. Changing hyperparameters
after inspecting 2025 would make it a development set; use fresh future data or
nested temporal evaluation for subsequent unbiased assessment. Validation
metrics are already affected by alpha selection and are not an untouched test.
Market comparisons remain retrospective because the raw lines lack timestamps.

New modeling modules are `models.py` (estimators/preprocessing), `backtesting.py`
(chronological selection/refit), `evaluation.py` (metrics/coverage), and
`model_reporting.py` (artifacts/provenance). Offline tests check fold boundaries,
train-only preprocessing, unseen/missing inputs, coherent outputs, price signs,
test-outcome perturbations, metric formulas, common cohorts and serialized-model
reproducibility.

## Joint distributions and rolling-origin validation

```bash
python scripts/validate_distributions.py --check-tests
```

This phase uses **nested chronological tuning**, rather than applying the alpha
selected in 2024 retrospectively to earlier forecasts. For each outer forecast
season, the mean model selects alpha using the latest earlier season, then refits
on all earlier seasons. The first fold has only 2021 available, so it tunes using
the first 75% versus last 25% of unique 2021 game dates. A date is never split
between training and validation. The alpha grid is the same fixed grid as the
baseline experiment; every fitting/tuning cutoff is saved.

| Forecast season | Mean fitting history | Residual-distribution fitting history | Role |
| --- | --- | --- | --- |
| 2022 | 2021 | None | Generate initial out-of-time residuals |
| 2023 | 2021–2022 | 2022 residual pairs | Development evaluation |
| 2024 | 2021–2023 | 2022–2023 residual pairs | Development evaluation |
| 2025 | 2021–2024 | 2022–2024 residual pairs | Retrospective audit |

Each model ultimately has **854 out-of-time home/away residual pairs** available
for calibration. No audit outcomes enter model fitting, calibration, or family
selection. The 2025 season was already inspected during baseline work, so it is
explicitly labeled a retrospective audit. These are retrospective experiments,
not a prospective study or a claim of a new untouched holdout.

The football, market-informed and direct-market means each get two candidate
joint residual distributions:

- **Paired bootstrap:** resample each historical home/away residual pair together,
  preserving empirical within-game dependence and tail shape.
- **Correlated Gaussian:** estimate residual means and a Ledoit–Wolf regularized
  covariance matrix from those same out-of-time pairs.

Both retain the measured residual bias. Per game, **10,000 draws** add fitted
residuals to that game's projected scores and censor negative team scores at
zero. Every draw derives total and margin from the sampled home/away scores.
Censoring changes the moments and creates mass at zero; all evaluation applies
to the final censored draws. Uncertainty is constant across matchups within a
fold; conditional volatility is not modeled yet.

Distribution family selection minimizes average **joint energy score** across
2023–2024 development forecasts; each development distribution was fitted only
on residuals from earlier seasons. Candidate families use the same evaluation
games. Selection is frozen before audit evaluation. The audit reports both
families and marks the one selected on development data.

Validation includes:

- CRPS for home score, away score, total and margin (exact empirical-ensemble
  calculation; lower is better).
- Joint home/away energy score (a Monte Carlo estimate using disjoint independent
  draw pairs; lower is better).
- Central 50%, 80% and 95% interval coverage, widths and interval scores.
- Descriptive Wilson intervals around coverage estimates.
- Randomized probability-integral-transform (PIT) histogram counts, including
  randomization at the zero-score atom.
- Sample correlation, zero-score mass and mean bias diagnostics.

Coverage near its nominal percentage is desirable but does not prove calibration.
Shared teams, multiple comparisons, limited development seasons and Monte Carlo
error constrain the interpretation. Scoring-rule definitions are documented by
[CRPS](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_ensemble.html)
and [energy score](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.es_ensemble.html);
the covariance estimator uses [Ledoit–Wolf shrinkage](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).

The build saves `artifacts/distributions/report.md`, per-game diagnostics,
aggregate metrics, PIT histogram counts, tuning provenance, all residuals,
distribution selection, and a fitted `models.joblib` bundle. Audit residuals are
saved for inspection but marked `calibration_eligible=False`. The final bundle
contains no audit residuals. `example_simulations.parquet` contains all 10,000
draws for one audit game from each of the three selected models.

To reproduce simulations using a trusted local bundle with `PYTHONPATH=src`:

```python
import joblib
import pandas as pd

models = joblib.load("artifacts/distributions/models.joblib")
games = pd.read_parquet("data/processed/game_modeling_dataset.parquet")
audit_games = games.loc[games.season == 2025]
for game_id, draws in models["ridge_football"].simulate(audit_games, draws=10_000):
    print(game_id, draws.mean())
```

The generator holds only one game's samples at a time. Per-game random streams
are deterministic across row order, batches and process restarts. Simulation
reads no outcome columns and rejects dates at or before either fitting cutoff.
The command reloads its newly saved bundle and reproduces all selected audit
sample means. `--draws`, `--seed`, `--first-season`, `--audit-season`, `--dataset`
and `--output-dir` are available CLI options.

New modules: `distributions.py` (joint residual models and simulation),
`distribution_evaluation.py` (scoring rules and calibration diagnostics),
`distribution_backtesting.py` (nested folds and family selection), and
`distribution_reporting.py` (artifacts and provenance). Tests verify fit dates,
paired dependence, analytical metric examples, output identities, deterministic
sampling, serialization, and invariance to changed audit outcomes.

## Discrete scores, settlement, and market probabilities

After generating the distribution artifacts:

```bash
python scripts/validate_market_probabilities.py --check-tests
```

The command reuses the saved nested out-of-time score forecasts; it does not
retrain their regressions. It verifies the dataset and model-bundle hashes,
checks cached game identities and dates, and checks that the saved mean models
reproduce audit forecasts. For each 2023–2025 forecast season, the discrete layer
fits **only earlier historical scores and earlier out-of-time residual pairs**.
2023–2024 are development diagnostics; 2025 remains a previously inspected
retrospective audit. No grid parameters are selected using its results.

The model assigns joint mass to integer home/away score pairs on a grid. A
correlated Gaussian residual density provides the broad shape. Learned local
score-frequency and symmetric margin-frequency corrections represent common
scoring patterns and key margins. The correction is the ratio of observed
counts to Gaussian-smoothed counts (bandwidth 2, one pseudocount, clipped to
0.25–4). These are fixed research assumptions, not tuned settings or an exact
possession-level scoring model. The Gaussian family is fixed for this layer;
it does not reuse a family selected in a later season for earlier folds.

Common-scoring support is 0 through 100, excluding 1. The rare defensive
one-point-safety final is not modeled; such historical scores are rejected.
The forecast rejects a grid whose unweighted Gaussian upper-tail bound exceeds
1e-6. Grid mass is normalized on supported nonnegative scores, without rounding
or censoring continuous draws. Corrections and normalization can change the
mean from the input point forecast; marginal volatility is not matchup-specific.

Regular-season tie odds are scaled to a Jeffreys-smoothed historical REG tie
rate, using only prior history and prior out-of-time means. Postseason terminal
tie mass is zero. The layer models **final scores including overtime**; it does
not separately simulate regulation, drives, or an additional overtime period.
Rule metadata covers only 2021–2025 and distinguishes the postseason change
in 2022 and regular-season change in 2025. Earlier pooled tie history cannot
establish calibration under the new 2025 REG rules.

See the official [2025 rulebook](https://operations.nfl.com/media/ntif5hxb/2025-nfl-rulebook-final.pdf),
[2024 rulebook](https://operations.nfl.com/media/24emxacq/2024-nfl-rulebook.pdf),
and [2023 record book](https://operations.nfl.com/media/nppjkdp1/2023-record-and-fact-book.pdf).

Settlement is an explicit full-game cash-bet contract:

| Market | Contract |
| --- | --- |
| Moneyline | Home or away wins; a final tie pushes by default. `--moneyline-tie loss` changes tied finals to losses. |
| Spread | `line` is the **home handicap**, even in an away-side row. Home -3 and a 24–21 final push both sides. The nflverse home-favored `market_spread_line` is negated exactly once. |
| Total | Over/under compare the sum of final scores with the threshold. An exact whole-number match pushes. |

Half-point lines cannot push. Quarter lines and regulation-only contracts are
rejected. This is not a reconstruction of every historical book's house rules;
voids, abandoned games and promotional payouts are not modeled. A two-way
push rule is consistent with the referenced
[general settlement rules](https://sportsbook.draftkings.com/help/general-betting-rules/general-rules).

Each market's win/loss/push probabilities are **exact sums of grid mass**.
Simulations use deterministic per-game random streams to sample the fitted grid.
Every draw's total and margin reconcile with its home/away integer scores.
The report command generates 10,000 draws for every audit game and compares
sample frequencies with exact probabilities; it saves complete samples for one
game per model, avoiding a large raw-simulation artifact.

Price comparison includes quoted implied probability, proportional two-way
no-vig probability, conditional model probability, edge versus the quoted
break-even price, and push-aware expected net profit. If `b` is profit per
unit stake on a win, then:

```text
conditional model probability = p_win / (p_win + p_loss)
expected net profit per unit = p_win * b - p_loss
```

The two team prices alone cannot identify draw mass when moneyline ties lose,
so that contract has no two-way no-vig moneyline comparator. Expected profit is
a snapshot diagnostic, not realized ROI or evidence of an executable edge.

Probability validation uses home moneyline, home spread and over total once per
game, avoiding double counting opposite sides. Binary Brier score/log loss and
ten-bin reliability tables condition on no push and exclude actual pushes.
Three-class win/loss/push Brier score (sum of squared errors, range 0–2), log
loss, and push calibration include pushes. The no-vig price baseline has no
inferred push mass, so its unconditional metrics stay missing. Tables show
sample sizes; missing quotes can leave different evaluation cohorts.

Artifacts in `artifacts/markets/`:

| File | Contents |
| --- | --- |
| `report.md` | Audit comparisons, chronological fitting boundaries, methods and limitations |
| `predictions.parquet` | Both sides, exact probabilities, settlement grades, prices, edge and expected profit |
| `metrics.parquet`, `calibration_bins.parquet` | Development/audit scoring and reliability diagnostics |
| `fit_log.parquet` | Calibration counts, dates, score-history size and tie parameters |
| `models.joblib` | Three discrete game models fitted before the audit |
| `simulation_checks.parquet` | Exact versus sampled rates for canonical markets in every audit game |
| `example_simulations.parquet` | One game's 10,000 draws per model |
| `run.json` | Input/code/artifact hashes, versions, assumptions and verification results |

The command reloads the bundle and verifies all audit market probabilities.
To query a trusted local bundle with `PYTHONPATH=src`:

```python
import joblib
import pandas as pd

models = joblib.load("artifacts/markets/models.joblib")
games = pd.read_parquet("data/processed/game_modeling_dataset.parquet")
audit_games = games.loc[games.season == 2025]
for game_id, grid in models["ridge_football"].grids(audit_games):
    print(game_id, grid.market_probabilities(market="moneyline", side="home"))
```

Modules: `discrete_scores.py`, `settlement.py`, `market_backtesting.py`,
`probability_evaluation.py`, and `market_reporting.py`. Tests include analytical
settlement/profit examples, sign conversion, whole/half-point lines, supported
scores, playoff winners, probabilities versus sample frequencies, chronology,
cached forecast integrity, audit-outcome perturbations and artifact reloads.

## Player datasets and opportunity models

Continue from the existing team and score artifacts:

```bash
python scripts/build_player_dataset.py
python scripts/train_opportunities.py
python scripts/validate_player_simulations.py
python scripts/verify_project.py
```

The player build caches 2021–2025 weekly player statistics, weekly rosters, and
PFR snap counts through nflreadpy. It produces **70,275 player forecast rows**
(67,427 named rows and 2,848 OTHER rows), plus observations, reconciliation
tables, identity exceptions, and `player_modeling_manifest.json` under
`data/processed/`. Eleven offensive statistics reconcile exactly with all 2,848
team-game totals. Raw snap coverage spans every completed team game.

Forecast candidates come from each team's **previous same-season game roster**
plus observed offensive participants in that earlier game. Current-game roster
status, participation, statistics and active-list information never decide who
gets a forecast. All prior roster statuses are retained, including zero-usage
players. The first game of each team-season has only OTHER; new/unlisted players'
current production is assigned to OTHER after forecasts. This avoids selecting
only the players who are known retrospectively to have played.

Player history uses prior 3/5 observed roster games and season averages within
player/team/season, including known zero usage. A trade starts a new team history.
Explicit inputs include prior snaps, offensive participation, attempts, carries,
targets, production, opportunity shares and efficiency. Current outcomes and
team exposure totals are separate labels, never predictive inputs.

Player ID joins use GSIS/PFR crosswalks without name guessing. The current build
records 55 offensive roster rows without GSIS IDs, 110 unidentified zero-offense
stat rows, 8 ambiguous PFR links, and 280 unmapped/ambiguous snap rows. These are
saved in `player_id_exceptions.parquet`. The 752 named forecast rows with unknown
offensive participation are excluded from participation scoring, not labeled
inactive. Positive observed opportunity establishes participation when snap
identity is unknown. Snapshot corrections and historical crosswalk revisions
remain a limitation even with correctly lagged calculations.

Data references: [nflreadpy loaders](https://nflreadpy.nflverse.com/api/load_functions/),
[snap definitions](https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html),
and [source availability/update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

Team-volume models forecast home/away pass attempts, carries, and sacks together;
total offensive plays are their sum. Ridge regularization is chosen within each
chronological fold using earlier validation games, against training-mean and
recent-average benchmarks. The player model fits an offensive-participation
logistic regression and conditional opportunity-share Poisson regressions.
Player regularization is fixed in advance. Allocation includes OTHER and sums
to each team's opportunity budget. Point shares normalize expected active
weights; simulation means can differ after stochastic participation/normalization.

Opportunity artifacts live in `artifacts/opportunities/`: fold models, predictions,
tuning, participation metrics, count metrics, source hashes, and `report.md`.
Scores cover all named candidates, players with positive prior usage, and OTHER
separately. The prior-usage cohort uses only earlier games. Audit opportunity
predictions use forecast team volume, never the observed test-game team totals.

## Correlated player production simulations

`validate_player_simulations.py` defaults to **10,000 draws per game** across
2023–2025 (855 games). 2022 supplies initial out-of-time calibration predictions.
Every later season's mean models and uncertainty estimates use earlier seasons
only. No setting is selected on the inspected 2025 audit.

The simulation follows these fitted components:

1. Sample integer final scores from the football model's discrete joint grid.
2. Sample six team-volume components conditional on those scores using an
   estimated joint score/volume residual covariance. Nonnegative counts use
   stochastic rounding; plays are the sum of attempts, carries and sacks.
3. Sample player offensive participation and allocate pass attempts, carries and
   receiving targets with conditional Dirichlet-multinomial shares. Concentration
   is fitted on earlier out-of-time weights conditional on earlier participation.
   Unknown-participation calibration groups are excluded.
4. Sample receptions using trained catch probabilities and earlier residual
   dispersion. Fit yards per reception/carry with exposure-weighted regression;
   earlier out-of-time errors estimate shared team and individual uncertainty.
   Yard totals remain signed integers and are bounded by 99 times opportunity.
5. Allocate completions, passing yards, and passing touchdowns consistently with
   the receiver totals. Historical offensive-TD counts conditional on final score
   determine a scoring budget; receiving/rushing TDs respect catch/carry capacities.

Every draw is checked for opportunity sums, completions equal to receptions,
passing yards equal to receiving yards, passing TDs equal to receiving TDs,
zero production for inactive players, and offensive TDs within the score budget.
This is an aggregate statistical simulation, not a possession/field-position
model. Passing lateral credits, QB rotations, injury events and individual
red-zone roles are approximated. Consistency is not proof of predictive skill.

Artifacts in `artifacts/players/` include the final `models.joblib`, all-fold
production predictions, simulation diagnostics, aggregate metrics, fitting dates,
randomized PIT histograms, a complete sample matchup, and `report.md`. The saved
model is reloaded and all example arrays are reproduced with actual outcomes
removed from the inputs. Per-game random streams make results reproducible.

Validation reports MAE/RMSE versus recent averages, CRPS, interval widths,
empirical interval coverage, **model-implied interval mass**, and randomized PIT.
The mass comparison matters because zero outcomes and discrete atoms can make
10th–90th percentile intervals contain more than 80% probability. Brier/log loss
use fixed research thresholds and a historical position-frequency benchmark;
these are **not historical sportsbook prop lines**.

The dashboard's prop calculator accepts an explicitly entered statistic, whole
or half line, American price and participation contract. It estimates win, loss,
push and void probabilities and computes expected net profit with both returned
stake outcomes handled. `all_candidates` counts nonparticipation as zero;
`offense_snap_required` voids without offensive participation. Neither should be
mistaken for a reconstruction of a sportsbook's any-snap rules.

Main modules: `player_dataset.py`, `opportunity_models.py`,
`opportunity_backtesting.py`, `player_production.py`, `player_simulation.py`,
`player_backtesting.py`, `player_props.py`, and `dashboard.py`.

## Archived quote replay and CLV

The empty template is `data/templates/player_quotes.csv`. Upload a completed
archive in the dashboard or run:

```bash
python scripts/backtest_player_quotes.py path/to/actual_quotes.csv
```

Required fields: `quote_id`, `game_id`, `team`, `player_id`, `book`, `quoted_at`,
`statistic`, `side`, `line`, `american_odds`, `participation`. Stake defaults to
one unit when its column is omitted. Timestamps require explicit offsets or `Z`.
The current bundle accepts completed 2025 NFL-season games and game-day quotes
strictly before kickoff. Earlier-week feature vintages are not reconstructed.
Kickoff uses the schedule's Eastern-time convention, including daylight saving:
[nflverse schedule definitions](https://github.com/nflverse/nfldata/blob/master/DATASETS.md).

Optional `closing_at`, `closing_line`, and `closing_odds` must appear together.
Closing snapshots must follow entry and precede kickoff. Positive line CLV means
a better entry threshold (lower for overs, higher for unders). Price CLV is the
closing-minus-entry raw implied probability, in percentage points, and is only
reported at the same line; it retains bookmaker margin. Uploaded closing quotes
are not independently verified as the final market price.

Replay prices and settles **every supplied quote** at its supplied stake. It
does not choose a strategy after observing returns. ROI is net profit divided by
all staked units, including stakes returned on pushes and voids. Unknown actual
participation prevents settlement of a contract requiring it. Results are saved
to `artifacts/quote_replay/` with input/model hashes. No real quote archive has
been supplied, so the project does not claim historical prop ROI or CLV results.

## Remaining work for live use

Connect player availability sources, persist timestamped odds/availability
vintages, and build future-game features before claiming live model pricing. The current rule
window and models cover 2021–2025; 2026 needs a separate data/rule review and
evaluation protocol. Improve calibration on chronological development data or
fresh future observations rather than tuning against the inspected 2025 audit.
