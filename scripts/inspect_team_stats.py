import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from nfl_model.data_loader import load_team_stats


team_stats = load_team_stats([2024, 2025])

print("\n=== TEAM STATS SHAPE ===")
print(team_stats.shape)

print("\n=== TEAM STATS COLUMNS ===")
for col in team_stats.columns:
    print(col)

print("\n=== FIRST 5 ROWS ===")
print(team_stats.head().to_string())