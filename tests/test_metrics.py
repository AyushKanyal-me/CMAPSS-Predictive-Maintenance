import numpy as np
import pytest

from src.evaluation.metrics import mae, nasa_score, regression_metrics, rmse


def test_perfect_predictions_have_zero_error_metrics() -> None:
    y_true = np.array([20.0, 10.0, 5.0], dtype=np.float64)
    y_pred = y_true.copy()

    assert rmse(y_true, y_pred) == 0.0
    assert mae(y_true, y_pred) == 0.0
    assert nasa_score(y_true, y_pred) == 0.0


def test_nasa_score_penalizes_late_predictions_more_than_early() -> None:
    y_true = np.array([50.0], dtype=np.float64)
    early_prediction = np.array([45.0], dtype=np.float64)
    late_prediction = np.array([55.0], dtype=np.float64)

    early_score = nasa_score(y_true, early_prediction)
    late_score = nasa_score(y_true, late_prediction)

    assert late_score > early_score


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        rmse(np.array([1.0, 2.0]), np.array([1.0]))


def test_regression_metrics_bundle_contains_standard_keys() -> None:
    metrics = regression_metrics(np.array([3.0, 2.0]), np.array([2.5, 1.5]))

    assert set(metrics.keys()) == {"rmse", "mae", "nasa_score"}