import pandas as pd
import pytest

from nfl_model import data_loader


@pytest.mark.parametrize("name", ["schedules", "team_stats", "player_stats", "snap_counts", "weekly_rosters"])
def test_cache_reused_normalized_and_explicitly_refreshable(name, monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "DEFAULT_CACHE_DIR", tmp_path)
    calls = []

    def download(seasons):
        calls.append(seasons)
        return pd.DataFrame({"season": seasons, "revision": len(calls)})

    monkeypatch.setattr(data_loader.nfl, "load_rosters_weekly" if name == "weekly_rosters" else f"load_{name}", download)
    loader = getattr(data_loader, f"load_{name}")
    first = loader([2025, 2024, 2025])
    cached = loader([2024, 2025])
    pd.testing.assert_frame_equal(first, cached)
    assert len(calls) == 1
    assert (tmp_path / f"{name}_2024_2025.parquet").exists()
    refreshed = loader([2024, 2025], refresh=True)
    assert len(calls) == 2
    assert (refreshed.revision == 2).all()


@pytest.mark.parametrize("seasons", [[], [True], [2024.5], ["2025"]])
def test_invalid_seasons_rejected(seasons):
    with pytest.raises(ValueError, match="integer years"):
        data_loader.load_team_stats(seasons)
