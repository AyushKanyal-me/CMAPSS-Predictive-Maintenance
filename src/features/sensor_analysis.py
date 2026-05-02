"""Sensor variability diagnostics used for EDA-based feature selection."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def get_sensor_columns(df: pd.DataFrame, prefix: str = "sensor_") -> list[str]:
    """Return sensor columns sorted by sensor index."""
    sensor_cols = [col for col in df.columns if col.startswith(prefix)]
    if not sensor_cols:
        raise ValueError(f"No columns found with prefix '{prefix}'.")

    def _sensor_key(name: str) -> int:
        parts = name.split("_")
        return int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 10**9

    return sorted(sensor_cols, key=_sensor_key)


def compute_sensor_summary(
    df: pd.DataFrame,
    sensor_columns: Sequence[str] | None = None,
    unit_col: str = "unit_id",
) -> pd.DataFrame:
    """Compute global and per-unit variability summaries for sensors."""
    if sensor_columns is None:
        sensor_columns = get_sensor_columns(df)
    else:
        sensor_columns = list(sensor_columns)

    missing = [col for col in sensor_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")
    if unit_col not in df.columns:
        raise ValueError(f"Missing unit column '{unit_col}'.")

    sensor_df = df[sensor_columns]
    summary = pd.DataFrame(index=sensor_columns)
    summary["global_mean"] = sensor_df.mean(axis=0)
    summary["global_std"] = sensor_df.std(axis=0, ddof=0)
    summary["global_var"] = sensor_df.var(axis=0, ddof=0)

    q1 = sensor_df.quantile(0.25, axis=0)
    q3 = sensor_df.quantile(0.75, axis=0)
    summary["global_iqr"] = q3 - q1

    per_unit_std = df.groupby(unit_col, sort=False)[sensor_columns].std(ddof=0).fillna(0.0)
    summary["mean_unit_std"] = per_unit_std.mean(axis=0)
    summary["median_unit_std"] = per_unit_std.median(axis=0)
    summary["max_unit_std"] = per_unit_std.max(axis=0)

    return summary.sort_values("mean_unit_std", ascending=True)


def suggest_low_variance_sensors(
    sensor_summary: pd.DataFrame,
    metric: str = "mean_unit_std",
    threshold: float = 0.05,
) -> list[str]:
    """Return sensor names with variability at or below the given threshold."""
    if metric not in sensor_summary.columns:
        raise ValueError(f"Metric '{metric}' was not found in sensor_summary columns.")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    keep_mask = sensor_summary[metric] <= threshold
    return sensor_summary.index[keep_mask].tolist()
