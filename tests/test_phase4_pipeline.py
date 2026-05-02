from pathlib import Path

import pytest

from src.pipeline.multi_dataset_benchmark import collect_available_datasets, parse_dataset_ids


def _touch_dataset_files(raw_dir: Path, dataset_id: str) -> None:
    (raw_dir / f"train_{dataset_id}.txt").write_text("", encoding="utf-8")
    (raw_dir / f"test_{dataset_id}.txt").write_text("", encoding="utf-8")
    (raw_dir / f"RUL_{dataset_id}.txt").write_text("", encoding="utf-8")


def test_parse_dataset_ids_normalizes_and_deduplicates() -> None:
    parsed = parse_dataset_ids("fd001,FD002,1,FD001")
    assert parsed == ["FD001", "FD002"]


def test_collect_available_datasets_skips_missing_when_enabled(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    _touch_dataset_files(raw_dir, "FD001")

    available, missing = collect_available_datasets(
        raw_data_dir=raw_dir,
        requested_datasets=["FD001", "FD002"],
        skip_missing=True,
    )

    assert available == ["FD001"]
    assert "FD002" in missing
    assert len(missing["FD002"]) == 3


def test_collect_available_datasets_raises_when_skip_missing_disabled(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    _touch_dataset_files(raw_dir, "FD001")

    with pytest.raises(FileNotFoundError, match="FD002"):
        collect_available_datasets(
            raw_data_dir=raw_dir,
            requested_datasets=["FD001", "FD002"],
            skip_missing=False,
        )