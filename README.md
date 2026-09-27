# NFL Lab

Streamlit NFL analytics app with historical game/player forecasts and a current
odds workspace. This repository includes the saved models and datasets needed
for all seven dashboard views. No training or data download is required to open it.

## Put these files on GitHub

1. Create a GitHub repository.
2. Upload the **contents of this folder** into the repository, preserving all
   subfolders. `app.py` and `requirements.txt` must be at the repository root.
3. Include `src/`, `data/`, `artifacts/`, `.streamlit/`, `.gitignore`, and both
   requirements files. The Python scripts, tests and documentation can be
   uploaded as well; everything in this prepared folder is intended for GitHub.
4. The ZIP is a transport copy. Extract it first; uploading only the ZIP does
   not deploy the app. On macOS, Command-Shift-period shows hidden files such as
   `.streamlit` and `.gitignore` in Finder.

The included `.gitignore` explicitly allows the runtime datasets/models while
excluding newly generated caches, training outputs, environments and secrets.
Do not replace it with the original development folder's blanket data exclusions.

## Host on Streamlit Community Cloud

GitHub stores the project; Streamlit Community Cloud runs the Python app.
GitHub Pages cannot run this Streamlit server.

1. Visit https://share.streamlit.io/ and choose **Create app**.
2. Select your GitHub repository and branch (usually `main`).
3. Set the main file path to **app.py**.
4. In **Advanced settings**, choose **Python 3.14**, matching the tested models.
5. Deploy and check the build logs. The host will provide a `streamlit.app` URL.

`requirements.txt` loads the pinned packages in `requirements.lock.txt`.
`.streamlit/config.toml` omits the development-only loopback address.
The bundle has been checked locally; a Linux cloud build has not yet been run.

Official deployment guide:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Season stats through 2026

Open **Season stats** in the sidebar. The default season is 2026; schedules,
team totals, player totals and game logs are available for 2021–2026. The
initial 2026 snapshot has statistics for 33 games and 2,294 player-game rows.
Filter by season, phase, week, team, player and position, or download CSV tables.

The data asset `data/season_stats/stats.zip` must stay zipped inside the
repository; it is different from the outer GitHub upload ZIP. Its source and
build timestamps appear in the app. Refresh current stats locally with:

```bash
python scripts/refresh_season_stats.py --seasons 2026 --refresh
```

Then upload the changed `data/season_stats/stats.zip` to GitHub. Other seasons
are preserved. This view shows observed stats; saved model forecasts still
cover 2025 and have not been retrained on 2026 outcomes.

## Optional live odds API

Manual quotes and CSV import work without any API account. For provider quotes,
open **Live odds & lines > API connection** and enter your own The Odds API key.
Alternatively, add this in your hosting app's **Secrets** settings:

```toml
ODDS_API_KEY = "your-key-here"
```

For local use, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and put your key in the copy. Never upload a real key
or secrets file to GitHub. No credential is included in this bundle.
API refreshes run only when clicked and use the provider account's quota.
Account access still needs verification with your own key.

## Run locally

Use Python 3.14, then run from this folder:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Scope and verification

- Historical forecasts cover 2025, using models fitted through 2024.
- The live odds board accepts manual, CSV, and provider snapshots. It has quote
  age indicators and price comparisons, but no current-game model forecasts.
- Quotes remain in each browser session. Export the history before leaving.
- In-play markets and automatic background collection are not supported.
- Project test results are in `docs/PROJECT_VERIFICATION.md`. The packaged
  app is checked separately; see `docs/DEPLOYMENT_CHECKS.json`.
- `FILE_MANIFEST.json` lists packaged files and their SHA-256 checksums.

See [modeling methods and rebuild instructions](docs/MODELING.md). Training
scripts and tests are included for maintenance. Full training caches and large
research outputs are excluded from this hosting copy and can be regenerated.
