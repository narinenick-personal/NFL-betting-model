import pandas as pd
import pytest
import numpy as np

from nfl_model.modeling_dataset import feature_groups


@pytest.fixture
def sample_data():
    # A changes home/away, has a bye, reaches the postseason, then starts a new season.
    records = [
        ("g1", 2024, 1, "2024-09-01", "A", "B", 10, 20, "REG"),
        ("g2", 2024, 2, "2024-09-08", "C", "A", 30, 40, "REG"),
        ("g3", 2024, 4, "2024-09-22", "A", "C", 50, 60, "REG"),
        ("g4", 2024, 5, "2024-09-29", "B", "A", 70, 80, "REG"),
        ("g5", 2024, 6, "2024-10-06", "A", "B", 90, 100, "REG"),
        ("g6", 2024, 7, "2024-10-13", "C", "A", 110, 120, "REG"),
        ("g7", 2024, 19, "2025-01-12", "A", "C", 21, 14, "WC"),
        ("g8", 2025, 1, "2025-09-01", "A", "B", 17, 7, "REG"),
    ]
    schedules = pd.DataFrame(records, columns=["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score", "game_type"])
    schedules = schedules.assign(location="Home", home_rest=7, away_rest=8, div_game=0, roof="outdoors", surface="grass", temp=60, wind=10, home_moneyline=-110, away_moneyline=100, spread_line=3, total_line=44.5)
    schedules.loc[schedules["game_id"].eq("g7"), "location"] = "Neutral"
    stats = []
    for row in schedules.itertuples(index=False):
        for team, opponent in ((row.home_team, row.away_team), (row.away_team, row.home_team)):
            stats.append({
                "game_id": row.game_id, "team": team, "opponent_team": opponent,
                "season": row.season, "week": row.week,
                "season_type": "REG" if row.game_type == "REG" else "POST",
                "attempts": 30, "carries": 20, "sacks_suffered": 2,
                "passing_yards": 240, "rushing_yards": 100,
                "passing_epa": 9, "rushing_epa": -2, "passing_cpoe": 4,
                "passing_interceptions": 1, "fumbles_lost_total": 4,
                "sack_fumbles_lost": 1, "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 1,
                "passing_first_downs": 12, "rushing_first_downs": 5,
                "passing_20": 3, "rushing_10": 4,
                "passing_tds": 2, "rushing_tds": 1,
                "penalties": 6, "penalty_yards": 45,
                "def_sacks": 3, "def_interceptions": 2, "def_qb_hits": 6,
            })
    return pd.DataFrame(stats), schedules


@pytest.fixture
def model_games():
    rng = np.random.default_rng(121)
    records = []
    for season in range(2021, 2026):
        for week in range(1, 9):
            row = {col: rng.normal() for col in feature_groups()["football_features"]}
            row.update({
                "game_id": f"{season}_{week:02d}_A_B", "season": season,
                "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
                "week": week, "home_team": "B", "away_team": "A",
                "game_type": "REG", "surface": "grass" if week % 2 else "fieldturf",
                "home_score": float(20 + week + rng.integers(0, 6)),
                "away_score": float(15 + week + rng.integers(0, 6)),
                "home_pregame_points_for_last_5": float(21 + week),
                "away_pregame_points_against_last_5": float(23 + week),
                "away_pregame_points_for_last_5": float(16 + week),
                "home_pregame_points_against_last_5": float(18 + week),
                "market_total_line": float(38 + 2 * week), "market_spread_line": 5.0,
                "market_home_moneyline": -150.0, "market_away_moneyline": 130.0,
                "market_home_spread_odds": -110.0, "market_away_spread_odds": -110.0,
                "market_under_odds": -110.0, "market_over_odds": -110.0,
                "temp": 60, "wind": 8, "roof": "outdoors",
            })
            row["game_total"] = row["home_score"] + row["away_score"]
            row["home_margin"] = row["home_score"] - row["away_score"]
            records.append(row)
    return pd.DataFrame(records)
