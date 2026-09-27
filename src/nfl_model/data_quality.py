from __future__ import annotations

import pandas as pd


def summarize(df: pd.DataFrame, name: str) -> dict:
    """Return lightweight data-quality information for a dataframe."""
    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    }


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> None:
    """Raise a useful error if expected columns are absent."""
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing expected columns: {', '.join(missing)}"
        )


def require_unique_keys(
    df: pd.DataFrame, keys: list[str], dataset_name: str
) -> None:
    """Reject missing or duplicate identifiers instead of silently dropping rows."""
    require_columns(df, keys, dataset_name)
    if df[keys].isna().any().any():
        raise ValueError(f"{dataset_name} has null keys: {keys}")
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, keys].head(5).to_dict("records")
        raise ValueError(f"{dataset_name} has duplicate keys {keys}: {examples}")
