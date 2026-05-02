"""Metrics for RUL regression benchmarking."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_1d_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Input values must not be empty.")
    return array


def _validated_targets(
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = _to_1d_array(y_true)
    pred = _to_1d_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. Got {true.shape} and {pred.shape}."
        )
    return true, pred


def rmse(y_true: Iterable[float] | np.ndarray, y_pred: Iterable[float] | np.ndarray) -> float:
    """Compute root mean squared error."""
    true, pred = _validated_targets(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(y_true: Iterable[float] | np.ndarray, y_pred: Iterable[float] | np.ndarray) -> float:
    """Compute mean absolute error."""
    true, pred = _validated_targets(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def nasa_score(y_true: Iterable[float] | np.ndarray, y_pred: Iterable[float] | np.ndarray) -> float:
    """Compute the NASA C-MAPSS asymmetric scoring metric.

    The score penalizes late predictions (predicting too much life remaining) more
    heavily than early predictions.
    """
    true, pred = _validated_targets(y_true, y_pred)
    diff = pred - true
    penalties = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(penalties))


def regression_metrics(
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
) -> dict[str, float]:
    """Return a standard metric bundle for RUL tasks."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }