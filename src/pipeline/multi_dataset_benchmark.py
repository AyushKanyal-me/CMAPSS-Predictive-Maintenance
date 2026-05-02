"""Phase 4 multi-dataset benchmark runner for NASA C-MAPSS FD001-FD004."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import torch

from src.data.constants import RECOMMENDED_DROPPED_SENSORS
from src.data.ingestion import load_fd_split, normalize_fd_id
from src.data.rul import add_rul_targets
from src.features.feature_engineering import build_rolling_features, create_sequence_dataset
from src.features.sensor_analysis import get_sensor_columns
from src.models.baseline import BaselineTrainingResult, split_by_unit, train_random_forest_baseline
from src.models.lstm import LSTMTrainingResult, train_lstm_regressor


FD_SPLITS = ("train", "test", "rul")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration bundle for phase 4 benchmarking."""

    raw_data_dir: Path
    output_dir: Path
    datasets: list[str]
    skip_missing: bool
    safe_mode: bool
    include_lstm: bool
    max_cpu_threads: int
    rul_cap: int
    validation_fraction: float
    random_state: int
    rolling_windows: tuple[int, ...]
    lag_steps: tuple[int, ...]
    sequence_length: int
    sequence_stride: int
    rf_n_estimators: int
    rf_n_jobs: int
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_dropout: float
    lstm_learning_rate: float
    lstm_batch_size: int
    lstm_epochs: int
    lstm_patience: int
    lstm_device: str
    log_to_mlflow: bool
    mlflow_tracking_uri: str | None
    mlflow_experiment: str
    log_models: bool


@dataclass
class DatasetArtifacts:
    """In-memory artifacts from one dataset benchmark run."""

    dataset_id: str
    baseline: BaselineTrainingResult
    lstm: LSTMTrainingResult | None
    train_units: int
    validation_units: int
    rolling_train_rows: int
    rolling_validation_rows: int
    sequence_train_samples: int | None
    sequence_validation_samples: int | None


def parse_dataset_ids(value: str) -> list[str]:
    """Parse and normalize comma-separated dataset ids."""
    if not value.strip():
        raise ValueError("datasets must not be empty.")

    parsed: list[str] = []
    for token in value.split(","):
        fd_id = normalize_fd_id(token.strip())
        if fd_id not in parsed:
            parsed.append(fd_id)
    return parsed


def _required_raw_paths(raw_data_dir: Path, dataset_id: str) -> dict[str, Path]:
    return {
        "train": raw_data_dir / f"train_{dataset_id}.txt",
        "test": raw_data_dir / f"test_{dataset_id}.txt",
        "rul": raw_data_dir / f"RUL_{dataset_id}.txt",
    }


def collect_available_datasets(
    raw_data_dir: Path,
    requested_datasets: Sequence[str],
    skip_missing: bool,
) -> tuple[list[str], dict[str, list[str]]]:
    """Return available datasets and per-dataset missing file paths."""
    available: list[str] = []
    missing_by_dataset: dict[str, list[str]] = {}

    for dataset_id in requested_datasets:
        paths = _required_raw_paths(raw_data_dir, dataset_id)
        missing_paths = [str(path) for path in paths.values() if not path.exists()]

        if missing_paths:
            missing_by_dataset[dataset_id] = missing_paths
            if not skip_missing:
                raise FileNotFoundError(
                    f"Missing required raw files for {dataset_id}: {missing_paths}. "
                    f"Set --skip-missing to continue with available datasets."
                )
            continue

        available.append(dataset_id)

    return available, missing_by_dataset


def _prepare_training_inputs(dataset_id: str, config: BenchmarkConfig) -> dict[str, Any]:
    train_df = load_fd_split(config.raw_data_dir, dataset_id, "train")
    train_df = add_rul_targets(train_df, cap=config.rul_cap, keep_raw=True)

    drop_sensors = set(RECOMMENDED_DROPPED_SENSORS)
    sensor_cols = [col for col in get_sensor_columns(train_df) if col not in drop_sensors]

    rolling_df = build_rolling_features(
        train_df,
        sensor_columns=sensor_cols,
        windows=config.rolling_windows,
        lags=config.lag_steps,
        min_periods=1,
        drop_na=True,
        include_original_sensors=False,
    )
    rolling_feature_cols = [col for col in rolling_df.columns if "_roll_" in col or "_lag_" in col]

    train_roll_df, validation_roll_df, validation_units = split_by_unit(
        rolling_df,
        validation_fraction=config.validation_fraction,
        random_state=config.random_state,
    )

    sequence_bundle: dict[str, Any] | None = None
    if config.include_lstm:
        sequence_features = ["op_setting_1", "op_setting_2", "op_setting_3", *sensor_cols]
        x_seq, y_seq, seq_index = create_sequence_dataset(
            train_df,
            feature_columns=sequence_features,
            target_col="rul",
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
        )
        validation_mask = seq_index["unit_id"].isin(set(validation_units)).to_numpy()
        sequence_bundle = {
            "x_train": x_seq[~validation_mask],
            "y_train": y_seq[~validation_mask],
            "x_validation": x_seq[validation_mask],
            "y_validation": y_seq[validation_mask],
        }

    return {
        "train_roll_df": train_roll_df,
        "validation_roll_df": validation_roll_df,
        "rolling_feature_cols": rolling_feature_cols,
        "validation_units": validation_units,
        "sequence_bundle": sequence_bundle,
    }


def run_single_dataset(dataset_id: str, config: BenchmarkConfig) -> DatasetArtifacts:
    """Train configured models for one dataset id."""
    prepared = _prepare_training_inputs(dataset_id, config)

    train_roll_df: pd.DataFrame = prepared["train_roll_df"]
    validation_roll_df: pd.DataFrame = prepared["validation_roll_df"]
    rolling_feature_cols: list[str] = prepared["rolling_feature_cols"]
    validation_units = prepared["validation_units"]

    baseline_result = train_random_forest_baseline(
        train_df=train_roll_df,
        validation_df=validation_roll_df,
        feature_columns=rolling_feature_cols,
        target_col="rul",
        random_state=config.random_state,
        n_estimators=config.rf_n_estimators,
        max_depth=None,
        n_jobs=config.rf_n_jobs,
    )

    lstm_result: LSTMTrainingResult | None = None
    sequence_train_samples: int | None = None
    sequence_validation_samples: int | None = None

    sequence_bundle = prepared["sequence_bundle"]
    if sequence_bundle is not None:
        lstm_result = train_lstm_regressor(
            x_train=sequence_bundle["x_train"],
            y_train=sequence_bundle["y_train"],
            x_validation=sequence_bundle["x_validation"],
            y_validation=sequence_bundle["y_validation"],
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            learning_rate=config.lstm_learning_rate,
            batch_size=config.lstm_batch_size,
            epochs=config.lstm_epochs,
            patience=config.lstm_patience,
            random_state=config.random_state,
            device=config.lstm_device,
        )
        sequence_train_samples = int(sequence_bundle["x_train"].shape[0])
        sequence_validation_samples = int(sequence_bundle["x_validation"].shape[0])

    return DatasetArtifacts(
        dataset_id=dataset_id,
        baseline=baseline_result,
        lstm=lstm_result,
        train_units=int(train_roll_df["unit_id"].nunique()),
        validation_units=int(len(validation_units)),
        rolling_train_rows=int(len(train_roll_df)),
        rolling_validation_rows=int(len(validation_roll_df)),
        sequence_train_samples=sequence_train_samples,
        sequence_validation_samples=sequence_validation_samples,
    )


def _rows_from_artifacts(artifacts: DatasetArtifacts) -> list[dict[str, Any]]:
    rows = [
        {
            "dataset_id": artifacts.dataset_id,
            "model": "random_forest",
            "rmse": artifacts.baseline.validation_metrics["rmse"],
            "mae": artifacts.baseline.validation_metrics["mae"],
            "nasa_score": artifacts.baseline.validation_metrics["nasa_score"],
            "loss": None,
            "best_epoch": None,
            "train_units": artifacts.train_units,
            "validation_units": artifacts.validation_units,
            "train_rows": artifacts.rolling_train_rows,
            "validation_rows": artifacts.rolling_validation_rows,
            "train_sequence_samples": artifacts.sequence_train_samples,
            "validation_sequence_samples": artifacts.sequence_validation_samples,
        }
    ]

    if artifacts.lstm is not None:
        rows.append(
            {
                "dataset_id": artifacts.dataset_id,
                "model": "lstm",
                "rmse": artifacts.lstm.best_validation_metrics["rmse"],
                "mae": artifacts.lstm.best_validation_metrics["mae"],
                "nasa_score": artifacts.lstm.best_validation_metrics["nasa_score"],
                "loss": artifacts.lstm.best_validation_metrics.get("loss"),
                "best_epoch": artifacts.lstm.best_epoch,
                "train_units": artifacts.train_units,
                "validation_units": artifacts.validation_units,
                "train_rows": artifacts.rolling_train_rows,
                "validation_rows": artifacts.rolling_validation_rows,
                "train_sequence_samples": artifacts.sequence_train_samples,
                "validation_sequence_samples": artifacts.sequence_validation_samples,
            }
        )

    return rows


def _log_to_mlflow(artifacts: DatasetArtifacts, config: BenchmarkConfig) -> None:
    if not config.log_to_mlflow:
        return

    with mlflow.start_run(run_name=f"{artifacts.dataset_id}_phase4", nested=True):
        mlflow.log_param("dataset_id", artifacts.dataset_id)
        mlflow.log_param("safe_mode", int(config.safe_mode))
        mlflow.log_param("include_lstm", int(config.include_lstm))
        mlflow.log_param("rul_cap", config.rul_cap)
        mlflow.log_param("rolling_windows", ",".join(map(str, config.rolling_windows)))
        mlflow.log_param("lag_steps", ",".join(map(str, config.lag_steps)))
        mlflow.log_param("validation_fraction", config.validation_fraction)
        mlflow.log_param("rf_n_estimators", config.rf_n_estimators)
        mlflow.log_param("rf_n_jobs", config.rf_n_jobs)

        mlflow.log_metric("train_units", float(artifacts.train_units))
        mlflow.log_metric("validation_units", float(artifacts.validation_units))
        mlflow.log_metric("baseline_val_rmse", artifacts.baseline.validation_metrics["rmse"])
        mlflow.log_metric("baseline_val_mae", artifacts.baseline.validation_metrics["mae"])
        mlflow.log_metric("baseline_val_nasa_score", artifacts.baseline.validation_metrics["nasa_score"])

        if config.log_models:
            mlflow.sklearn.log_model(
                artifacts.baseline.model,
                name=f"{artifacts.dataset_id.lower()}_random_forest",
            )

        if artifacts.lstm is not None:
            mlflow.log_param("sequence_length", config.sequence_length)
            mlflow.log_param("sequence_stride", config.sequence_stride)
            mlflow.log_param("lstm_hidden_size", config.lstm_hidden_size)
            mlflow.log_param("lstm_num_layers", config.lstm_num_layers)
            mlflow.log_param("lstm_dropout", config.lstm_dropout)
            mlflow.log_param("lstm_learning_rate", config.lstm_learning_rate)
            mlflow.log_param("lstm_batch_size", config.lstm_batch_size)
            mlflow.log_param("lstm_epochs", config.lstm_epochs)
            mlflow.log_param("lstm_patience", config.lstm_patience)
            mlflow.log_param("lstm_device", config.lstm_device)

            mlflow.log_metric("lstm_best_epoch", float(artifacts.lstm.best_epoch))
            mlflow.log_metric("lstm_val_rmse", artifacts.lstm.best_validation_metrics["rmse"])
            mlflow.log_metric("lstm_val_mae", artifacts.lstm.best_validation_metrics["mae"])
            mlflow.log_metric("lstm_val_nasa_score", artifacts.lstm.best_validation_metrics["nasa_score"])
            if "loss" in artifacts.lstm.best_validation_metrics:
                mlflow.log_metric("lstm_val_loss", float(artifacts.lstm.best_validation_metrics["loss"]))

            if config.log_models:
                mlflow.pytorch.log_model(
                    artifacts.lstm.model.cpu(),
                    name=f"{artifacts.dataset_id.lower()}_lstm",
                )


def _write_reports(
    output_dir: Path,
    config: BenchmarkConfig,
    available_datasets: Sequence[str],
    missing_by_dataset: dict[str, list[str]],
    summary_rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "phase4_multi_dataset_summary.csv"
    summary_json = output_dir / "phase4_multi_dataset_summary.json"
    run_meta_json = output_dir / "phase4_run_metadata.json"

    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    run_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "raw_data_dir": str(config.raw_data_dir),
            "output_dir": str(config.output_dir),
        },
        "available_datasets": list(available_datasets),
        "missing_by_dataset": missing_by_dataset,
    }
    run_meta_json.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return summary_csv, summary_json, run_meta_json


def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    """Execute benchmark for requested datasets and return summary DataFrame."""
    if config.max_cpu_threads > 0:
        torch.set_num_threads(config.max_cpu_threads)

    available_datasets, missing_by_dataset = collect_available_datasets(
        raw_data_dir=config.raw_data_dir,
        requested_datasets=config.datasets,
        skip_missing=config.skip_missing,
    )
    if not available_datasets:
        missing_text = json.dumps(missing_by_dataset, indent=2)
        raise FileNotFoundError(
            "No requested datasets are available in data/raw. "
            f"Missing files by dataset:\n{missing_text}"
        )

    if config.include_lstm and config.lstm_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "lstm_device='cuda' requested but CUDA is not available on this machine. "
            "Use --lstm-device cpu locally or run the same command on Colab GPU."
        )

    if config.log_to_mlflow:
        if config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)

    summary_rows: list[dict[str, Any]] = []

    if config.log_to_mlflow:
        parent_name = f"phase4_multi_dataset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        parent_context = mlflow.start_run(run_name=parent_name)
    else:
        parent_context = None

    try:
        for dataset_id in available_datasets:
            artifacts = run_single_dataset(dataset_id, config)
            summary_rows.extend(_rows_from_artifacts(artifacts))
            _log_to_mlflow(artifacts, config)

        summary_csv, summary_json, run_meta_json = _write_reports(
            output_dir=config.output_dir,
            config=config,
            available_datasets=available_datasets,
            missing_by_dataset=missing_by_dataset,
            summary_rows=summary_rows,
        )

        print("Phase 4 benchmark completed.")
        print("Available datasets:", available_datasets)
        if missing_by_dataset:
            print("Skipped datasets with missing files:")
            print(json.dumps(missing_by_dataset, indent=2))
        print("Saved:", summary_csv)
        print("Saved:", summary_json)
        print("Saved:", run_meta_json)

        if config.log_to_mlflow:
            active = mlflow.active_run()
            if active is not None:
                print("MLflow parent run_id:", active.info.run_id)
            print("MLflow tracking URI:", mlflow.get_tracking_uri())
            print("MLflow experiment:", config.mlflow_experiment)

        summary_df = pd.DataFrame(summary_rows)
        print("Validation summary:")
        print(summary_df.sort_values(["dataset_id", "rmse"]).reset_index(drop=True))
        return summary_df
    finally:
        if parent_context is not None:
            parent_context.__exit__(None, None, None)


def _build_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    safe_mode = args.safe_mode
    max_cpu_threads = args.max_cpu_threads

    rf_n_estimators = args.rf_n_estimators
    rf_n_jobs = args.rf_n_jobs
    lstm_batch_size = args.lstm_batch_size
    lstm_epochs = args.lstm_epochs
    lstm_patience = args.lstm_patience

    if safe_mode:
        if rf_n_estimators is None:
            rf_n_estimators = 250
        if rf_n_jobs is None:
            rf_n_jobs = max_cpu_threads if max_cpu_threads > 0 else 2
        if lstm_batch_size is None:
            lstm_batch_size = 128
        if lstm_epochs is None:
            lstm_epochs = 8
        if lstm_patience is None:
            lstm_patience = 3
    else:
        if rf_n_estimators is None:
            rf_n_estimators = 500
        if rf_n_jobs is None:
            rf_n_jobs = -1
        if lstm_batch_size is None:
            lstm_batch_size = 256
        if lstm_epochs is None:
            lstm_epochs = 20
        if lstm_patience is None:
            lstm_patience = 4

    return BenchmarkConfig(
        raw_data_dir=args.raw_data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        datasets=parse_dataset_ids(args.datasets),
        skip_missing=args.skip_missing,
        safe_mode=safe_mode,
        include_lstm=args.include_lstm,
        max_cpu_threads=max_cpu_threads,
        rul_cap=args.rul_cap,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
        rolling_windows=tuple(int(value) for value in args.rolling_windows.split(",")),
        lag_steps=tuple(int(value) for value in args.lag_steps.split(",")),
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        rf_n_estimators=rf_n_estimators,
        rf_n_jobs=rf_n_jobs,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        lstm_dropout=args.lstm_dropout,
        lstm_learning_rate=args.lstm_learning_rate,
        lstm_batch_size=lstm_batch_size,
        lstm_epochs=lstm_epochs,
        lstm_patience=lstm_patience,
        lstm_device=args.lstm_device,
        log_to_mlflow=args.log_to_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        log_models=args.log_models,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase 4 multi-dataset C-MAPSS benchmark.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing train_FD00x.txt, test_FD00x.txt, and RUL_FD00x.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where benchmark summary reports will be saved.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="FD001,FD002,FD003,FD004",
        help="Comma-separated dataset ids (for example: FD001,FD003).",
    )
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip datasets missing raw files instead of failing.",
    )
    parser.add_argument(
        "--safe-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use conservative local-machine defaults to reduce resource usage.",
    )
    parser.add_argument(
        "--include-lstm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train LSTM in addition to RandomForest baseline.",
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=2,
        help="Thread cap for safe local execution. Use 0 to keep library defaults.",
    )
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rolling-windows", type=str, default="5,15,30")
    parser.add_argument("--lag-steps", type=str, default="1,3,5")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--rf-n-estimators", type=int)
    parser.add_argument("--rf-n-jobs", type=int)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument("--lstm-learning-rate", type=float, default=1e-3)
    parser.add_argument("--lstm-batch-size", type=int)
    parser.add_argument("--lstm-epochs", type=int)
    parser.add_argument("--lstm-patience", type=int)
    parser.add_argument("--lstm-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--log-to-mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log runs to MLflow.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Optional explicit MLflow tracking URI.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="cmapss_phase4_multi_dataset",
        help="MLflow experiment name for phase 4 benchmarking.",
    )
    parser.add_argument(
        "--log-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log trained models to MLflow (disabled by default for faster runs).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = _build_config_from_args(args)
    run_benchmark(config)


if __name__ == "__main__":
    main()