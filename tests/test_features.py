import pandas as pd

from nfl_model.features import (
    add_team_rolling_features,
    add_team_season_averages,
)


def test_rolling_features_do_not_use_current_game():
    df = pd.DataFrame({
        "team": ["A", "A", "A"],
        "season": [2025, 2025, 2025],
        "week": [1, 2, 3],
        "points": [20, 30, 40],
    })

    result = add_team_rolling_features(df, window=2)

    assert pd.isna(result.loc[0, "points_rolling_2"])
    assert result.loc[1, "points_rolling_2"] == 20
    assert result.loc[2, "points_rolling_2"] == 25


def test_season_average_does_not_use_current_game():
    df = pd.DataFrame({
        "team": ["A", "A", "A"],
        "season": [2025, 2025, 2025],
        "week": [1, 2, 3],
        "points": [20, 30, 40],
    })

    result = add_team_season_averages(df)

    assert pd.isna(result.loc[0, "points_season_avg"])
    assert result.loc[1, "points_season_avg"] == 20
    assert result.loc[2, "points_season_avg"] == 25