import numpy as np
import pandas as pd
import torch

from src.models.baseline import split_by_unit
from src.models.lstm import LSTMRegressor, train_lstm_regressor


def test_split_by_unit_creates_disjoint_unit_sets() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "cycle": [1, 2] * 5,
            "feature_1": np.arange(10, dtype=np.float32),
            "rul": np.arange(10, 0, -1, dtype=np.int32),
        }
    )

    train_df, validation_df, validation_units = split_by_unit(
        df,
        validation_fraction=0.4,
        random_state=7,
    )

    assert set(train_df["unit_id"]).isdisjoint(set(validation_df["unit_id"]))
    assert len(validation_units) == 2


def test_lstm_regressor_forward_output_shape() -> None:
    model = LSTMRegressor(input_size=3, hidden_size=16, num_layers=1, dropout=0.0)
    batch = torch.randn(4, 5, 3, dtype=torch.float32)

    output = model(batch)

    assert output.shape == (4,)


def test_train_lstm_regressor_returns_history_and_predictions() -> None:
    rng = np.random.default_rng(seed=123)
    x_train = rng.normal(size=(12, 4, 2)).astype(np.float32)
    y_train = rng.normal(size=(12,)).astype(np.float32)
    x_validation = rng.normal(size=(6, 4, 2)).astype(np.float32)
    y_validation = rng.normal(size=(6,)).astype(np.float32)

    result = train_lstm_regressor(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=4,
        epochs=3,
        patience=0,
        random_state=123,
    )

    assert not result.history.empty
    assert result.best_epoch >= 1
    assert result.validation_predictions.shape == (len(y_validation),)