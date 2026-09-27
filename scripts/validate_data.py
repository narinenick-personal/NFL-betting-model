from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_model.data_loader import (  # noqa: E402
    load_players,
    load_player_stats,
    load_schedules,
    load_team_stats,
    load_weekly_rosters,
)
from nfl_model.data_quality import require_columns, summarize  # noqa: E402


SEASONS = [2024, 2025]


def main() -> None:
    print("Loading schedules...")
    schedules = load_schedules(SEASONS)

    print("Loading player stats...")
    player_stats = load_player_stats(SEASONS)

    print("Loading team stats...")
    team_stats = load_team_stats(SEASONS)

    print("Loading players...")
    players = load_players()

    print("Loading weekly rosters...")
    rosters = load_weekly_rosters(SEASONS)

    require_columns(
        schedules,
        ["season", "week", "game_id", "home_team", "away_team"],
        "schedules",
    )

    require_columns(
        player_stats,
        ["season", "week", "player_id"],
        "player_stats",
    )

    require_columns(
        team_stats,
        ["season", "week", "team"],
        "team_stats",
    )

    require_columns(
        players,
        ["gsis_id", "display_name"],
        "players",
    )

    require_columns(
        rosters,
        ["season", "week", "gsis_id"],
        "weekly_rosters",
    )

    datasets = [
        ("schedules", schedules),
        ("player_stats", player_stats),
        ("team_stats", team_stats),
        ("players", players),
        ("weekly_rosters", rosters),
    ]

    print("\nDATA VALIDATION")
    print("-" * 60)

    for name, df in datasets:
        summary = summarize(df, name)
        print(
            f"{name:18} "
            f"rows={summary['rows']:,} "
            f"cols={summary['columns']:,} "
            f"duplicates={summary['duplicate_rows']:,} "
            f"missing={summary['missing_cells']:,}"
        )

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    main()
