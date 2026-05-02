"""FastAPI app for serving baseline RUL predictions."""

from __future__ import annotations

import os
import shutil
import urllib.request
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.constants import RECOMMENDED_DROPPED_SENSORS, SENSOR_COLUMNS
from src.features.feature_engineering import build_rolling_features

DEFAULT_MODEL_PATH = Path("models/fd001_random_forest_baseline.joblib")
DEFAULT_ROLLING_WINDOWS = (5, 15, 30)
DEFAULT_LAG_STEPS = (1, 3, 5)

MODEL_NAME = os.getenv("CMAPSS_MODEL_NAME", "fd001_random_forest_baseline")
MODEL_PATH = Path(os.getenv("CMAPSS_BASELINE_MODEL", str(DEFAULT_MODEL_PATH)))
MODEL_URL = os.getenv("CMAPSS_BASELINE_MODEL_URL", "")


def _parse_int_tuple(value: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return fallback
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return fallback
    return tuple(int(item) for item in items)


ROLLING_WINDOWS = _parse_int_tuple(os.getenv("CMAPSS_ROLLING_WINDOWS", ""), DEFAULT_ROLLING_WINDOWS)
LAG_STEPS = _parse_int_tuple(os.getenv("CMAPSS_LAG_STEPS", ""), DEFAULT_LAG_STEPS)


class SensorRecord(BaseModel):
    model_config = {"extra": "forbid"}

    cycle: int = Field(..., ge=1)
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    model_config = {"extra": "forbid"}

    unit_id: int = Field(..., ge=1)
    records: list[SensorRecord] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    unit_id: int
    cycle: int
    model: str
    rul: float
    feature_count: int
    rolling_windows: tuple[int, ...]
    lag_steps: tuple[int, ...]


app = FastAPI(title="C-MAPSS RUL API", version="0.1.0")


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        if MODEL_URL:
            _download_model(MODEL_URL, MODEL_PATH)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Baseline model not found. "
                f"Set CMAPSS_BASELINE_MODEL or CMAPSS_BASELINE_MODEL_URL. (Path: '{MODEL_PATH}')."
            )
    return joblib.load(MODEL_PATH)


def _download_model(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _records_to_frame(unit_id: int, records: list[SensorRecord]) -> pd.DataFrame:
    rows = [record.model_dump() for record in records]
    df = pd.DataFrame(rows)
    df["unit_id"] = unit_id

    required = ["cycle", "op_setting_1", "op_setting_2", "op_setting_3", *SENSOR_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = df.sort_values("cycle").drop_duplicates(subset=["cycle"], keep="last")
    return df.reset_index(drop=True)


def _build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    sensor_cols = [col for col in SENSOR_COLUMNS if col not in RECOMMENDED_DROPPED_SENSORS]

    rolling_df = build_rolling_features(
        df,
        sensor_columns=sensor_cols,
        windows=ROLLING_WINDOWS,
        lags=LAG_STEPS,
        min_periods=1,
        drop_na=True,
        include_original_sensors=False,
    )

    if rolling_df.empty:
        raise ValueError(
            "Not enough cycles to compute rolling features. "
            "Provide at least 6 cycles for lag=5, or more for stable windows."
        )

    rolling_feature_cols = [col for col in rolling_df.columns if "_roll_" in col or "_lag_" in col]
    if not rolling_feature_cols:
        raise ValueError("No rolling features were generated from the provided records.")

    latest_row = rolling_df.sort_values("cycle").iloc[-1]
    feature_frame = latest_row[rolling_feature_cols].to_frame().T
    return feature_frame, rolling_feature_cols, int(latest_row["cycle"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/baseline", response_model=PredictResponse)
def predict_baseline(request: PredictRequest) -> PredictResponse:
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    df = _records_to_frame(request.unit_id, request.records)

    try:
        feature_frame, feature_cols, cycle = _build_feature_frame(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and expected_features != len(feature_cols):
        raise HTTPException(
            status_code=500,
            detail=(
                "Feature count mismatch: "
                f"model expects {expected_features}, but request produced {len(feature_cols)}."
            ),
        )

    prediction = float(model.predict(feature_frame)[0])
    return PredictResponse(
        unit_id=request.unit_id,
        cycle=cycle,
        model=MODEL_NAME,
        rul=prediction,
        feature_count=len(feature_cols),
        rolling_windows=ROLLING_WINDOWS,
        lag_steps=LAG_STEPS,
    )
