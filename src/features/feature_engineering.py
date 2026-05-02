"""Feature engineering utilities for tabular and sequence modeling workflows."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

from .sensor_analysis import get_sensor_columns


def _require_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_rolling_features(
    df: pd.DataFrame,
    sensor_columns: Sequence[str] | None = None,
    windows: Sequence[int] = (5, 15, 30),
    lags: Sequence[int] = (1, 3, 5),
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    min_periods: int = 1,
    drop_na: bool = True,
    include_original_sensors: bool = False,
) -> pd.DataFrame:
    """Create rolling mean/std and lag features for each sensor column.

    Rolling operations are computed independently per unit to prevent data leakage
    across engines.
    """
    if sensor_columns is None:
        sensor_columns = get_sensor_columns(df)
    else:
        sensor_columns = list(sensor_columns)

    windows = tuple(int(window) for window in windows)
    lags = tuple(int(lag) for lag in lags)

    if not windows:
        raise ValueError("windows must contain at least one positive integer.")
    if any(window <= 0 for window in windows):
        raise ValueError("All rolling windows must be positive integers.")
    if any(lag <= 0 for lag in lags):
        raise ValueError("All lag values must be positive integers.")
    if min_periods <= 0:
        raise ValueError("min_periods must be >= 1.")

    _require_columns(df, [unit_col, cycle_col, *sensor_columns])

    out = df.sort_values([unit_col, cycle_col]).copy()
    grouped = out.groupby(unit_col, sort=False)

    engineered_cols: list[str] = []
    feature_frames: list[pd.DataFrame] = []
    for sensor_col in sensor_columns:
        sensor_group = grouped[sensor_col]
        sensor_features: dict[str, pd.Series] = {}

        for window in windows:
            mean_col = f"{sensor_col}_roll_mean_w{window}"
            std_col = f"{sensor_col}_roll_std_w{window}"

            sensor_features[mean_col] = sensor_group.transform(
                lambda values: values.rolling(window=window, min_periods=min_periods).mean()
            )
            sensor_features[std_col] = sensor_group.transform(
                lambda values: values.rolling(window=window, min_periods=min_periods).std(ddof=0)
            ).fillna(0.0)

        for lag in lags:
            lag_col = f"{sensor_col}_lag_{lag}"
            sensor_features[lag_col] = sensor_group.shift(lag)

        if sensor_features:
            feature_frames.append(pd.DataFrame(sensor_features, index=out.index))
            engineered_cols.extend(sensor_features.keys())

    if feature_frames:
        # Concatenate features in bulk to avoid highly fragmented frames.
        out = pd.concat([out, *feature_frames], axis=1)

    if drop_na and engineered_cols:
        out = out.dropna(subset=engineered_cols)

    if not include_original_sensors:
        out = out.drop(columns=sensor_columns)

    return out.reset_index(drop=True)


def create_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "rul",
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    sequence_length: int = 30,
    stride: int = 1,
    dtype: np.dtype | str = np.float32,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate per-unit sliding-window sequences for PyTorch models.

    Returns:
        X: ndarray with shape (samples, sequence_length, n_features)
        y: ndarray with shape (samples,)
        index_df: metadata for each sequence target point (unit_id and cycle)
    """
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer.")
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    _require_columns(df, [unit_col, cycle_col, target_col, *feature_columns])

    sorted_df = df.sort_values([unit_col, cycle_col]).copy()

    if sorted_df[feature_columns].isna().any().any():
        raise ValueError("feature_columns contain NaN values. Clean features before sequence creation.")

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    sequence_units: list[int] = []
    sequence_cycles: list[int] = []

    for unit_id, unit_frame in sorted_df.groupby(unit_col, sort=False):
        feature_values = unit_frame[feature_columns].to_numpy(dtype=dtype, copy=False)
        target_values = unit_frame[target_col].to_numpy(dtype=np.float32, copy=False)
        cycle_values = unit_frame[cycle_col].to_numpy(copy=False)

        n_rows = len(unit_frame)
        if n_rows < sequence_length:
            continue

        for start_idx in range(0, n_rows - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            sequences.append(feature_values[start_idx:end_idx])
            targets.append(float(target_values[end_idx - 1]))
            sequence_units.append(int(unit_id))
            sequence_cycles.append(int(cycle_values[end_idx - 1]))

    if not sequences:
        raise ValueError("No sequences were generated. Reduce sequence_length or verify input data.")

    x_array = np.stack(sequences).astype(dtype, copy=False)
    y_array = np.asarray(targets, dtype=np.float32)
    index_df = pd.DataFrame(
        {
            unit_col: sequence_units,
            cycle_col: sequence_cycles,
            target_col: y_array,
        }
    )
    return x_array, y_array, index_df


def dataframe_fingerprint(df: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Return a deterministic SHA256 hash of a DataFrame slice for version tracking."""
    if columns is None:
        frame = df
    else:
        columns = list(columns)
        _require_columns(df, columns)
        frame = df[columns]

    row_hashes = pd.util.hash_pandas_object(frame, index=True)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()