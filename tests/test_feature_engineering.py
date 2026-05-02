import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    build_rolling_features,
    create_sequence_dataset,
    dataframe_fingerprint,
)


def test_build_rolling_features_respects_unit_boundaries() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "sensor_1": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "rul": [3, 2, 1, 3, 2, 1],
        }
    )

    features = build_rolling_features(
        df,
        sensor_columns=["sensor_1"],
        windows=[2],
        lags=[1],
        min_periods=1,
        drop_na=False,
        include_original_sensors=True,
    )

    unit_2_first = features[(features["unit_id"] == 2) & (features["cycle"] == 1)].iloc[0]
    assert unit_2_first["sensor_1_roll_mean_w2"] == 100.0
    assert unit_2_first["sensor_1_roll_std_w2"] == 0.0
    assert np.isnan(unit_2_first["sensor_1_lag_1"])


def test_create_sequence_dataset_does_not_cross_units() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "sensor_1": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
            "sensor_2": [4.0, 5.0, 6.0, 14.0, 15.0, 16.0],
            "rul": [2, 1, 0, 2, 1, 0],
        }
    )

    x_array, y_array, index_df = create_sequence_dataset(
        df,
        feature_columns=["sensor_1", "sensor_2"],
        target_col="rul",
        sequence_length=2,
        stride=1,
    )

    assert x_array.shape == (4, 2, 2)
    assert y_array.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert index_df["unit_id"].tolist() == [1, 1, 2, 2]
    assert index_df["cycle"].tolist() == [2, 3, 2, 3]

    # The first sequence of unit 2 starts with sensor_1=11.0, proving no cross-unit splice.
    assert x_array[2, 0, 0] == 11.0


def test_dataframe_fingerprint_changes_when_data_changes() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1],
            "cycle": [1, 2],
            "rul": [2, 1],
        }
    )

    hash_1 = dataframe_fingerprint(df)
    hash_2 = dataframe_fingerprint(df.copy())
    assert hash_1 == hash_2

    updated = df.copy()
    updated.loc[1, "rul"] = 0
    hash_3 = dataframe_fingerprint(updated)
    assert hash_3 != hash_1