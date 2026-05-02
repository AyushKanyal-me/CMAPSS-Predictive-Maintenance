# NASA C-MAPSS Predictive Maintenance - Implementation Plan

This plan outlines the strategy for building a high-impact predictive maintenance portfolio project using the NASA C-MAPSS dataset. The focus is on demonstrating professional engineering rigor, clear decision-making, and modern MLOps practices.

## Core Project Decisions

- **Framework**: PyTorch (Modern, industry-standard, and clear for sequence modeling).
- **Tracking**: MLflow (Comprehensive logging of parameters, metrics, and models).
- **Compute Strategy**: Jupyter Notebooks (`.ipynb`) optimized for Google Colab to handle intensive training and large-scale data processing.
- **Dataset Strategy**: Modular approach targeting individual sub-datasets (FD001-FD004) first with segregated logic, followed by an integrated pipeline for the full dataset.
- **Documentation**: All critical engineering decisions (RUL capping, sensor selection, etc.) will be documented directly in the `README.md` for maximum visibility to recruiters.

## Proposed Structure

The project is located at: `/Users/ayushkanyal/Desktop/Predictive Maintainence`

### 1. Project Scaffolding (Completed)

- `data/raw/` & `data/processed/`
- `notebooks/`: Primary workspace for EDA and Training.
- `src/`: Modular code for data ingestion, feature engineering, and evaluation.
- `models/`: Artifact storage.
- `docs/`: Supplementary documentation.
- `tests/`: Validation for core logic.

### 2. Implementation Phases

**Phase 1: Data Pipeline & EDA (Colab-Ready)**
- **Objective**: Establish the ingestion pipeline for individual FD00x files.
- **Status**: Completed (23 April 2026).
- **Tasks**:
  - Build `01_data_ingestion_and_eda.ipynb`.
  - Implement sensor analysis to justify dropping constant/noisy columns (e.g., 1, 5, 6, 10, 16, 18, 19).
  - Implement Piecewise Linear RUL construction with configurable capping (defaulting to ~125 cycles).

**Phase 1 Deliverables Added**
- `src/data/ingestion.py`: FD00x file normalization and loading helpers.
- `src/data/rul.py`: Raw + capped RUL target builders for train and test splits.
- `src/features/sensor_analysis.py`: Sensor variability summaries and low-variance sensor suggestions.
- `notebooks/01_data_ingestion_and_eda.ipynb`: End-to-end ingestion and EDA workflow for FD001.
- `tests/test_rul.py`: Unit tests for RUL capping behavior and test-truth integration.

**Phase 2: Feature Engineering & Tracking**
- **Objective**: Create training sequences and integrate experiment tracking.
- **Status**: Completed (24 April 2026).
- **Tasks**:
  - [x] Build `02_feature_engineering.ipynb`.
  - [x] Implement rolling window transformations (mean, std, lag) for traditional ML.
  - [x] Implement 3D sequence generators `(samples, time_steps, features)` for PyTorch.
  - [x] Integrate MLflow to log data versioning and preprocessing parameters.

**Phase 3: Modeling (Baseline & Deep Learning)**
- **Objective**: Compare traditional vs. sequence models.
- **Status**: Completed for FD001 single-dataset workflow (27 April 2026).
- **Tasks**:
  - [x] Build `03_training_and_evaluation.ipynb`.
  - [x] Train a baseline (Random Forest) on rolling features.
  - [x] Build a PyTorch LSTM for sequence-to-RUL prediction.
  - [x] Log all experiments to MLflow.

**Phase 4: Integration & Portfolio Finalization**
- **Objective**: Combine sub-datasets and finalize documentation.
- **Status**: Completed (02 May 2026; full FD001-FD004 baseline run captured).
- **Tasks**:
  - [x] Create a master script/notebook to run the pipeline across all 4 datasets.
  - [x] Finalize `README.md` with the "Decision Log" section highlighting:
    - Why RUL capping matters.
    - Why specific sensors were dropped based on variance.
    - Why RMSE vs. MAE impacts maintenance scheduling decisions.

## Verification Plan

### Automated Tests
- Unit tests in `tests/` for RUL calculation to prevent "future leak" in piecewise capping.
- Validation tests for sequence generation to ensure engine boundaries are respected.

### Manual Verification
- Review MLflow dashboard to compare model performance across different RUL caps.
- Verify that notebooks run seamlessly in Google Colab (including dependency installs).

## Execution Tracker (24 April 2026)

- [x] Dependencies installed in `.venv` from `requirements.txt`.
- [x] FD001 raw files added to `data/raw/` (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`).
- [x] `notebooks/01_data_ingestion_and_eda.ipynb` executed end-to-end without errors.
- [x] Processed output generated: `data/processed/train_FD001_with_rul_cap_125.parquet`.
- [x] Phase 2 notebook added: `notebooks/02_feature_engineering.ipynb`.
- [x] Rolling feature dataset generated: `data/processed/train_FD001_rolling_features.parquet`.
- [x] Sequence artifacts generated: `data/processed/train_FD001_sequences_x_t30.npy`, `data/processed/train_FD001_sequences_y_t30.npy`, `data/processed/train_FD001_sequence_index_t30.parquet`.
- [x] MLflow run logged for Phase 2 preprocessing (`experiment`: `cmapss_phase2_feature_engineering`).

## Execution Tracker (27 April 2026)

- [x] Phase 3 model/evaluation modules added: `src/models/baseline.py`, `src/models/lstm.py`, `src/evaluation/metrics.py`.
- [x] Phase 3 notebook added and executed end-to-end: `notebooks/03_training_and_evaluation.ipynb`.
- [x] Safe execution profile used (CPU-only LSTM, capped threads, reduced training intensity).
- [x] Validation metrics recorded (FD001, unit-wise split):
  - Random Forest: RMSE `16.6912`, MAE `11.7403`, NASA score `33798.2982`.
  - LSTM: RMSE `41.4496`, MAE `36.5466`, NASA score `591225.3451`.
- [x] Model artifacts saved to `models/`:
  - `fd001_random_forest_baseline.joblib`
  - `fd001_lstm_regressor.pt`
  - `fd001_phase3_validation_metrics.csv`
- [x] MLflow Phase 3 run logged (`experiment`: `cmapss_phase3_modeling`, `run_id`: `f0aa0549e4b549a0b34856f5b5c3e72b`).

## Execution Tracker (27 April 2026 - Phase 4)

- [x] Phase 4 master benchmark script added: `src/pipeline/multi_dataset_benchmark.py`.
- [x] Pipeline package exports added: `src/pipeline/__init__.py`.
- [x] Phase 4 unit tests added: `tests/test_phase4_pipeline.py`.
- [x] README updated with phase 4 runner commands (safe local + Colab GPU variants).
- [x] Colab handoff documentation and results template added for GPU-only runs.
- [x] Smoke baseline run saved to `reports/smoke_phase4` (FD001 only).

## Execution Tracker (02 May 2026 - Phase 4 Full Run)

- [x] Full FD001-FD004 baseline benchmark completed with safe defaults.
- [x] Phase 4 reports updated in `reports/`:
  - `phase4_multi_dataset_summary.csv`
  - `phase4_multi_dataset_summary.json`
  - `phase4_run_metadata.json`
- [x] Validation metrics recorded (Random Forest baseline):
  - FD001: RMSE `16.6639`, MAE `11.7373`, NASA score `33564.97`.
  - FD002: RMSE `19.5947`, MAE `15.0507`, NASA score `144504.91`.
  - FD003: RMSE `13.3285`, MAE `8.7833`, NASA score `27589.47`.
  - FD004: RMSE `20.1623`, MAE `14.2500`, NASA score `394994.36`.
