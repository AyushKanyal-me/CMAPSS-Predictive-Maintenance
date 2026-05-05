# NASA C-MAPSS Predictive Maintenance

End-to-end predictive maintenance system for turbofan engines using the NASA C-MAPSS dataset, built with production-ready feature engineering, leakage-safe evaluation, and a deployed FastAPI inference service.

| Result set | Dataset | RMSE | MAE | NASA Score | Live API |
| --- | --- | --- | --- | --- | --- |
| Random Forest pipeline (safe profile) | FD001 | 16.66 | 11.74 | 33564.97 | https://cmapss-predictive-maintenance.onrender.com/docs |
| Random Forest pipeline (safe profile) | FD002 | 19.59 | 15.05 | 144504.91 | https://cmapss-predictive-maintenance.onrender.com/docs |
| Random Forest pipeline (safe profile) | FD003 | 13.33 | 8.78 | 27589.47 | https://cmapss-predictive-maintenance.onrender.com/docs |
| Random Forest pipeline (safe profile) | FD004 | 20.16 | 14.25 | 394994.36 | https://cmapss-predictive-maintenance.onrender.com/docs |

## Highlights

- Rolling statistics and lag features engineered per engine to avoid data leakage.
- Random Forest pipeline deployed behind a FastAPI endpoint and Dockerized for cloud hosting.
- Multi-dataset benchmark runner with consistent settings and report artifacts.
- MLflow tracking and pytest coverage for reproducibility.

## Engineering Decisions

### 1. Piecewise RUL capping
I cap Remaining Useful Life at 125 cycles rather than letting it decline linearly from the start. Early cycles represent a healthy engine, so high-variance RUL targets are not meaningful. The cap keeps training focused on the degradation phase and improves near-failure accuracy.

### 2. Sensor selection
I drop sensors 1, 5, 6, 10, 16, 18, and 19 because they are near-constant across the lifecycle. Removing them reduces noise and discourages overfitting to simulator artifacts.

### 3. Evaluation aligned to maintenance risk
I train with RMSE but report the NASA scoring function because late predictions are far more costly than early ones in maintenance planning. This keeps evaluation aligned with operational risk.

### 4. Leakage-safe validation
I split by engine ID rather than shuffling rows. Each engine in the validation set is unseen during training, which preserves temporal integrity.

## Reproducibility

I log experiments with MLflow and track model performance across datasets. Unit tests cover feature engineering, metrics, and pipeline behavior to keep regressions out of the training and scoring flow.

## Project Structure

```text
├── data/               # Raw and processed datasets
├── notebooks/          # EDA and training notebooks
├── src/                # Pipeline source code
│   ├── data/           # Ingestion and RUL logic
│   ├── features/       # Rolling and lag feature engineering
│   ├── models/         # Random Forest and LSTM training code
│   └── evaluation/     # Metrics and scoring
├── reports/            # Benchmark outputs and metadata
└── models/             # Local model artifacts (downloaded in deployment)
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Setup

Download the NASA C-MAPSS dataset:
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

Notes:
- The archive contains a nested CMAPSSData.zip.
- Extract the inner zip and place the raw text files in data/raw.

Minimum required files for FD001:
- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

For full Phase 4 coverage, add FD002 to FD004 equivalents.

## Local API

Run the FastAPI server:

```bash
uvicorn src.api.app:app --reload --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict/baseline \
  -H "Content-Type: application/json" \
  --data-binary @docs/api_sample_request.json
```

Notes:
- Provide at least 6 cycles to satisfy lag features (1, 3, 5).
- For stable rolling statistics, send 30 or more cycles (window size 30).
- Override defaults with `CMAPSS_BASELINE_MODEL`, `CMAPSS_BASELINE_MODEL_URL`, `CMAPSS_ROLLING_WINDOWS`, and `CMAPSS_LAG_STEPS`.

## Docker

```bash
docker build -t cmapss-rul-api .
docker run -p 8000:8000 cmapss-rul-api
```

## Deployment

The container is ready for any Docker-compatible host. Render configuration lives in [render.yaml](render.yaml).

Render quickstart:
1. Push the repo to GitHub.
2. Create a Render Web Service connected to the repo.
3. Render detects the Dockerfile and exposes port 8000.
4. Verify the health endpoint at https://<your-service>/health.

## Benchmarks

Run the multi-dataset benchmark (Random Forest only):

```bash
python -m src.pipeline.multi_dataset_benchmark \
  --safe-mode \
  --datasets FD001,FD002,FD003,FD004 \
  --skip-missing
```

Outputs are written to the reports directory:
- [reports/phase4_multi_dataset_summary.csv](reports/phase4_multi_dataset_summary.csv)
- [reports/phase4_multi_dataset_summary.json](reports/phase4_multi_dataset_summary.json)
- [reports/phase4_run_metadata.json](reports/phase4_run_metadata.json)

## Tests

```bash
pytest
```

## Documentation and Reports

- Phase 3 notebook: [notebooks/03_training_and_evaluation.ipynb](notebooks/03_training_and_evaluation.ipynb)
- Phase 4 summary CSV: [reports/phase4_multi_dataset_summary.csv](reports/phase4_multi_dataset_summary.csv)
- Phase 4 summary JSON: [reports/phase4_multi_dataset_summary.json](reports/phase4_multi_dataset_summary.json)
- Phase 4 metadata: [reports/phase4_run_metadata.json](reports/phase4_run_metadata.json)
- Colab handoff notes: [docs/phase4_colab_gpu_handoff.md](docs/phase4_colab_gpu_handoff.md)
- Colab results template: [reports/colab_gpu_results.md](reports/colab_gpu_results.md)
- Portfolio checklist: [docs/portfolio_readiness.md](docs/portfolio_readiness.md)
