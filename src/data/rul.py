"""RUL target builders for train and test partitions."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_rul_targets(
    df: pd.DataFrame,
    cap: Optional[int] = 125,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    output_col: str = "rul",
    keep_raw: bool = True,
) -> pd.DataFrame:
    """Add raw and piecewise-capped RUL labels to a training DataFrame.

    Raw RUL is computed per unit as:
    max_cycle(unit) - current_cycle
    """
    _require_columns(df, [unit_col, cycle_col])

    if cap is not None and cap < 0:
        raise ValueError("cap must be >= 0 or None")

    max_cycle_per_unit = df.groupby(unit_col, sort=False)[cycle_col].transform("max")
    rul_raw = (max_cycle_per_unit - df[cycle_col]).astype(int)

    if (rul_raw < 0).any():
        raise ValueError("Computed negative RUL values; check cycle ordering and input data.")

    out = df.copy()
    if keep_raw:
        out[f"{output_col}_raw"] = rul_raw

    if cap is None:
        out[output_col] = rul_raw
    else:
        out[output_col] = rul_raw.clip(upper=cap).astype(int)

    return out


def add_test_rul_targets(
    test_df: pd.DataFrame,
    test_rul: pd.DataFrame | pd.Series,
    cap: Optional[int] = 125,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    output_col: str = "rul",
    keep_raw: bool = True,
) -> pd.DataFrame:
    """Attach RUL labels to the test split using the external NASA truth file."""
    _require_columns(test_df, [unit_col, cycle_col])

    if cap is not None and cap < 0:
        raise ValueError("cap must be >= 0 or None")

    if isinstance(test_rul, pd.Series):
        final_rul_by_unit = pd.Series(test_rul.values, index=range(1, len(test_rul) + 1))
    else:
        if {"unit_id", "final_rul"}.issubset(test_rul.columns):
            final_rul_by_unit = test_rul.set_index("unit_id")["final_rul"]
        elif test_rul.shape[1] == 1:
            only_col = test_rul.columns[0]
            final_rul_by_unit = pd.Series(test_rul[only_col].values, index=range(1, len(test_rul) + 1))
        else:
            raise ValueError(
                "test_rul must be a Series, a 1-column DataFrame, or include unit_id/final_rul columns."
            )

    max_cycle_per_unit = test_df.groupby(unit_col, sort=False)[cycle_col].transform("max")
    mapped_final_rul = test_df[unit_col].map(final_rul_by_unit)

    if mapped_final_rul.isna().any():
        missing_units = sorted(test_df.loc[mapped_final_rul.isna(), unit_col].unique())
        raise ValueError(f"Missing truth labels for unit ids: {missing_units}")

    rul_raw = (max_cycle_per_unit + mapped_final_rul - test_df[cycle_col]).astype(int)
    if (rul_raw < 0).any():
        raise ValueError("Computed negative RUL values for test split; verify truth labels.")

    out = test_df.copy()
    if keep_raw:
        out[f"{output_col}_raw"] = rul_raw

    if cap is None:
        out[output_col] = rul_raw
    else:
        out[output_col] = rul_raw.clip(upper=cap).astype(int)

    return out
