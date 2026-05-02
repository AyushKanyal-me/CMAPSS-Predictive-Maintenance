"""Baseline modeling utilities for tabular RUL prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..evaluation.metrics import regression_metrics


def _require_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_by_unit(
    df: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    unit_col: str = "unit_id",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Create train/validation splits by unit id to prevent leakage."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the open interval (0, 1).")

    _require_columns(df, [unit_col])

    unique_units = np.asarray(sorted(df[unit_col].unique()))
    if unique_units.size < 2:
        raise ValueError("At least two units are required for train/validation splitting.")

    validation_count = max(1, int(round(unique_units.size * validation_fraction)))
    validation_count = min(validation_count, unique_units.size - 1)

    rng = np.random.default_rng(seed=random_state)
    shuffled_units = unique_units.copy()
    rng.shuffle(shuffled_units)

    validation_units = np.sort(shuffled_units[:validation_count])
    is_validation = df[unit_col].isin(validation_units)

    train_df = df.loc[~is_validation].copy().reset_index(drop=True)
    validation_df = df.loc[is_validation].copy().reset_index(drop=True)
    return train_df, validation_df, validation_units


@dataclass
class BaselineTrainingResult:
    """Container for baseline model artifacts and summary metrics."""

    model: RandomForestRegressor
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    validation_predictions: np.ndarray


def train_random_forest_baseline(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "rul",
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int | None = None,
    n_jobs: int = -1,
    model_kwargs: dict[str, Any] | None = None,
) -> BaselineTrainingResult:
    """Train a RandomForest baseline and return train/validation metrics."""
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")

    _require_columns(train_df, [*feature_columns, target_col])
    _require_columns(validation_df, [*feature_columns, target_col])

    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    if model_kwargs:
        params.update(model_kwargs)

    model = RandomForestRegressor(**params)
    model.fit(train_df[feature_columns], train_df[target_col])

    train_predictions = model.predict(train_df[feature_columns])
    validation_predictions = model.predict(validation_df[feature_columns])

    train_metrics = regression_metrics(train_df[target_col].to_numpy(), train_predictions)
    validation_metrics = regression_metrics(validation_df[target_col].to_numpy(), validation_predictions)

    return BaselineTrainingResult(
        model=model,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        validation_predictions=np.asarray(validation_predictions, dtype=np.float32),
    )