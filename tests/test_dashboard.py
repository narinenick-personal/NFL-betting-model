from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from nfl_model.discrete_scores import ScoreGrid


class FixedScoreModel:
    def grids(self, games):
        for game_id in games.game_id:
            yield game_id, ScoreGrid(np.array([24, 20]), np.array([21, 20]), np.array([.75, .25]))


def test_dashboard_quote_controls_and_invalid_odds(tmp_path, model_games):
    data = tmp_path / "data/processed"
    models = tmp_path / "artifacts/markets"
    data.mkdir(parents=True)
    models.mkdir(parents=True)
    model_games.loc[model_games.season.eq(2025)].to_parquet(data / "game_modeling_dataset.parquet", index=False)
    joblib.dump({name: FixedScoreModel() for name in ("ridge_football", "ridge_market", "market_lines")}, models / "models.joblib")
    app = AppTest.from_string(f"from pathlib import Path\nfrom nfl_model.dashboard import render_dashboard\nrender_dashboard(Path({str(tmp_path)!r}))", default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "NFL Lab"
    app.selectbox(key="market").set_value("total").run()
    assert not app.exception
    odds = next(item for item in app.number_input if item.label == "American odds")
    odds.set_value(0).run()
    assert not app.exception and len(app.error) == 1
    app.radio(key="view").set_value("Player props").run()
    assert not app.exception
    assert "not ready" in app.info[0].value


def test_dashboard_handles_missing_artifacts(tmp_path):
    app = AppTest.from_string(f"from pathlib import Path\nfrom nfl_model.dashboard import render_dashboard\nrender_dashboard(Path({str(tmp_path)!r}))").run()
    assert not app.exception
    assert "Build the modeling datasets" in app.info[0].value
