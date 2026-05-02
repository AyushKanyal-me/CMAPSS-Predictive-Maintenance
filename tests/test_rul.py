import pandas as pd
import pytest

from src.data.rul import add_rul_targets, add_test_rul_targets


def test_add_rul_targets_computes_raw_and_capped_labels_without_reordering() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 5, 1, 2, 3],
        },
        index=[10, 11, 12, 13, 14, 20, 21, 22],
    )

    result = add_rul_targets(df, cap=2)

    assert result.index.tolist() == df.index.tolist()
    assert result["rul_raw"].tolist() == [4, 3, 2, 1, 0, 2, 1, 0]
    assert result["rul"].tolist() == [2, 2, 2, 1, 0, 2, 1, 0]


def test_add_rul_targets_without_cap_matches_raw_rul() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "cycle": [1, 2, 3],
        }
    )

    result = add_rul_targets(df, cap=None)
    assert result["rul_raw"].tolist() == [2, 1, 0]
    assert result["rul"].tolist() == [2, 1, 0]


def test_add_test_rul_targets_uses_truth_file_per_unit() -> None:
    test_df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        }
    )
    test_rul = pd.DataFrame({"final_rul": [2, 4]})

    result = add_test_rul_targets(test_df, test_rul, cap=4)

    assert result["rul_raw"].tolist() == [4, 3, 2, 5, 4]
    assert result["rul"].tolist() == [4, 3, 2, 4, 4]


def test_add_rul_targets_rejects_negative_cap() -> None:
    df = pd.DataFrame({"unit_id": [1], "cycle": [1]})
    with pytest.raises(ValueError, match="cap"):
        add_rul_targets(df, cap=-1)
