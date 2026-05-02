"""Ingestion helpers for NASA C-MAPSS FD00x text files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from .constants import C_MAPSS_COLUMNS

SplitName = Literal["train", "test", "rul"]


def normalize_fd_id(fd_id: int | str) -> str:
    """Normalize a dataset identifier to FD001..FD004 format."""
    if isinstance(fd_id, int):
        fd_num = fd_id
    else:
        candidate = str(fd_id).strip().upper()
        if candidate.startswith("FD"):
            candidate = candidate[2:]
        if not candidate.isdigit():
            raise ValueError(f"Invalid FD id '{fd_id}'. Expected one of FD001..FD004.")
        fd_num = int(candidate)

    if fd_num < 1 or fd_num > 4:
        raise ValueError(f"Invalid FD id '{fd_id}'. Expected one of FD001..FD004.")

    return f"FD{fd_num:03d}"


def _split_file_name(fd_id: str, split: SplitName) -> str:
    if split == "rul":
        return f"RUL_{fd_id}.txt"
    return f"{split}_{fd_id}.txt"


def read_cmapss_file(file_path: str | Path) -> pd.DataFrame:
    """Read a train/test C-MAPSS file into a typed DataFrame.

    The raw NASA files may contain irregular spacing and trailing delimiters,
    so the parser drops all-null columns after reading.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")

    expected_cols = len(C_MAPSS_COLUMNS)
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Unexpected number of columns in {path.name}. "
            f"Expected {expected_cols}, got {df.shape[1]}."
        )

    df.columns = C_MAPSS_COLUMNS
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def read_cmapss_rul_file(file_path: str | Path) -> pd.DataFrame:
    """Read an RUL truth file and attach unit ids in canonical order."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    rul_df = rul_df.dropna(axis=1, how="all")

    if rul_df.shape[1] != 1:
        raise ValueError(
            f"Unexpected number of columns in {path.name}. Expected 1, got {rul_df.shape[1]}."
        )

    rul_df.columns = ["final_rul"]
    rul_df["final_rul"] = rul_df["final_rul"].astype(int)
    rul_df.insert(0, "unit_id", range(1, len(rul_df) + 1))
    return rul_df


def load_fd_split(raw_data_dir: str | Path, fd_id: int | str, split: SplitName) -> pd.DataFrame:
    """Load a specific FD00x split from data/raw."""
    fd_name = normalize_fd_id(fd_id)
    path = Path(raw_data_dir) / _split_file_name(fd_name, split)

    if split == "rul":
        return read_cmapss_rul_file(path)
    return read_cmapss_file(path)


def load_fd_dataset(raw_data_dir: str | Path, fd_id: int | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/test/rul files for one FD00x dataset."""
    train_df = load_fd_split(raw_data_dir, fd_id, "train")
    test_df = load_fd_split(raw_data_dir, fd_id, "test")
    rul_df = load_fd_split(raw_data_dir, fd_id, "rul")
    return train_df, test_df, rul_df
