import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from nfl_model.data_loader import load_schedules


schedules = load_schedules([2024, 2025])

print("\n=== SCHEDULES SHAPE ===")
print(schedules.shape)

print("\n=== SCHEDULES COLUMNS ===")
for col in schedules.columns:
    print(col)

print("\n=== FIRST 5 ROWS ===")
print(schedules.head().to_string())