import pandas as pd
import pytest

from nfl_model.data_quality import require_columns, require_unique_keys


def test_require_columns_passes():
    df = pd.DataFrame({"season": [2025], "week": [1]})
    require_columns(df, ["season", "week"], "test")


def test_require_columns_fails():
    df = pd.DataFrame({"season": [2025]})

    with pytest.raises(ValueError):
        require_columns(df, ["season", "week"], "test")


def test_missing_identifiers_rejected():
    with pytest.raises(ValueError, match="null keys"):
        require_unique_keys(pd.DataFrame({"game_id": [None]}), ["game_id"], "test")
