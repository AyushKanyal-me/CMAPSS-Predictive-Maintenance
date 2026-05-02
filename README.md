# NASA C-MAPSS Predictive Maintenance 🚀

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Colab](https://img.shields.io/badge/Compute-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

An advanced predictive maintenance system for turbofan engines using the NASA C-MAPSS dataset. This project goes beyond basic regression to implement piecewise linear Remaining Useful Life (RUL) prediction, complex time-series feature engineering, and sequence-based Deep Learning models.

## 🧠 Engineering Decision Log (The "Why")

Standard tutorials often treat C-MAPSS as a simple regression problem. This project makes several non-obvious engineering decisions to reflect real-world industrial constraints.

### 1. Piecewise Linear RUL Capping (Default: 125 Cycles)
**Decision**: We cap the target RUL at 125 cycles instead of letting it decline linearly from the start.
**Rationale**: In early cycles, an engine is "healthy" and doesn't show degradation. Predicting a specific RUL of 300 vs 250 is physically meaningless when no fault has manifested. Capping the RUL forces the model to focus on the degradation phase, significantly improving the performance on critical "near-failure" predictions.

### 2. Feature Selection & Sensor Dropping
**Decision**: Dropped sensors 1, 5, 6, 10, 16, 18, and 19.
**Rationale**: These sensors show near-zero variance across the entire engine lifecycle (flat lines). Including them adds noise and increases the risk of overfitting to simulation artifacts rather than physical degradation patterns.

### 3. Metric Selection: RMSE + NASA Scoring Function
**Decision**: Use Root Mean Squared Error (RMSE) for training, but evaluate using the asymmetric NASA Scoring Function.
**Rationale**: In predictive maintenance, a **late prediction** (predicting more life than remains) is catastrophic, while an **early prediction** is merely a suboptimal maintenance cost. The NASA Scoring Function penalizes late predictions much more heavily than early ones.

### 4. Temporal Data Splitting
**Decision**: No random shuffling.
**Rationale**: Since this is time-series data, random shuffling leads to data leakage. We split by Engine ID, ensuring that engines in the test set are completely unseen during training.

---

## 🛠️ Project Structure

```text
├── data/               # Raw and processed datasets
├── notebooks/          # Colab-ready Jupyter notebooks for EDA and Training
├── src/                # Modular source code for the pipeline
│   ├── data/           # Ingestion logic
│   ├── features/       # Rolling windows and sequence generation
│   ├── models/         # PyTorch architectures (LSTM, 1D-CNN)
│   └── evaluation/     # Custom metrics and scoring
├── models/             # Saved model weights and MLflow artifacts
└── implementation_plan.md
```

## 🚀 Getting Started

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Ingestion**: Download the raw NASA files and place them in `data/raw` (see Data Setup).
3. **MLflow Tracking**: All experiments are logged locally or to a remote MLflow server.

## Data Setup

Download the NASA C-MAPSS dataset archive:
- https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

Extraction notes:
- The archive contains a nested `CMAPSSData.zip`.
- Extract the inner zip and place the raw text files in `data/raw/`.

Minimum required files for FD001:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

For full Phase 4 coverage, also add FD002-FD004 equivalents.

## Tests

```bash
pytest
```

## API + Docker Deployment

### Local API
Run the FastAPI server with the baseline Random Forest model:

```bash
uvicorn src.api.app:app --reload --port 8000
```

Example request (sample payload provided):

```bash
curl -X POST http://localhost:8000/predict/baseline \
   -H "Content-Type: application/json" \
   --data-binary @docs/api_sample_request.json
```

Notes:
- Provide at least 6 cycles to satisfy lag features (`1,3,5`).
- For stable rolling statistics, send 30+ cycles (window size `30`).
- Override defaults with `CMAPSS_BASELINE_MODEL`, `CMAPSS_ROLLING_WINDOWS`, and `CMAPSS_LAG_STEPS`.
- If the model file is not present, set `CMAPSS_BASELINE_MODEL_URL` to a GitHub Release asset URL.

### Docker
Build and run the containerized API:

```bash
docker build -t cmapss-rul-api .
docker run -p 8000:8000 cmapss-rul-api
```

### Hosting Online
Any Docker-compatible host works (Render, Railway, Fly.io, Cloud Run). Deploy the image and expose port `8000`.

Render quickstart:
1. Push the repo to GitHub.
2. In Render, create a new Web Service and connect the repo.
3. Render detects the Dockerfile. Keep `PORT=8000` (see `render.yaml`).
4. Deploy and test `https://<your-service>/health`.

`render.yaml` already sets `PORT=8000` and `healthCheckPath: /health` for the Docker deploy.

## 🧪 Phase 4 Multi-Dataset Benchmark

Use the phase 4 runner to benchmark FD001-FD004 with the same feature engineering and model settings.

### Safe local run (recommended on laptops/desktops)
```bash
python -m src.pipeline.multi_dataset_benchmark \
   --safe-mode \
   --datasets FD001,FD002,FD003,FD004 \
   --skip-missing
```

### Heavier run for Colab GPU
```bash
python -m src.pipeline.multi_dataset_benchmark \
   --safe-mode \
   --include-lstm \
   --lstm-device cuda \
   --datasets FD001,FD002,FD003,FD004 \
   --skip-missing
```

Outputs are written to `reports/`:
- `phase4_multi_dataset_summary.csv`
- `phase4_multi_dataset_summary.json`
- `phase4_run_metadata.json`

If your local machine is too slow for full LSTM benchmarking, use Colab GPU and drop results into `reports/colab_gpu_results.md` (template in `docs/phase4_colab_gpu_handoff.md`).

---

## ✅ Final Run Guide

Run the full multi-dataset benchmark (baseline only):

```bash
python -m src.pipeline.multi_dataset_benchmark \
   --safe-mode \
   --datasets FD001,FD002,FD003,FD004 \
   --skip-missing
```

Run the test suite:

```bash
pytest
```

Start the local API (baseline model):

```bash
uvicorn src.api.app:app --reload --port 8000
```

For Docker and Render deployment, see the API + Docker Deployment section above.

---

## 📊 Results Summary
FD001 Phase 3 run completed with a leakage-safe unit-wise split and MLflow tracking.

- **Baseline (Random Forest, safe profile)**: RMSE `16.69` | MAE `11.74` | NASA Score `33798.30`
- **LSTM (PyTorch, CPU safe profile)**: RMSE `41.45` | MAE `36.55` | NASA Score `591225.35` (best epoch: `8`)
- **Phase 4 full benchmark (Random Forest, safe profile)**:
   - FD001: RMSE `16.66` | MAE `11.74` | NASA Score `33564.97`
   - FD002: RMSE `19.59` | MAE `15.05` | NASA Score `144504.91`
   - FD003: RMSE `13.33` | MAE `8.78` | NASA Score `27589.47`
   - FD004: RMSE `20.16` | MAE `14.25` | NASA Score `394994.36`

MLflow tracking:
- Experiment: `cmapss_phase3_modeling`
- Run ID: `f0aa0549e4b549a0b34856f5b5c3e72b`

## Documentation and Reports

- Phase 3 notebook: [notebooks/03_training_and_evaluation.ipynb](notebooks/03_training_and_evaluation.ipynb)
- Phase 4 full summary: [reports/phase4_multi_dataset_summary.csv](reports/phase4_multi_dataset_summary.csv)
- Phase 4 full JSON: [reports/phase4_multi_dataset_summary.json](reports/phase4_multi_dataset_summary.json)
- Phase 4 run metadata: [reports/phase4_run_metadata.json](reports/phase4_run_metadata.json)
- Phase 4 smoke summary: [reports/smoke_phase4/phase4_multi_dataset_summary.csv](reports/smoke_phase4/phase4_multi_dataset_summary.csv)
- Phase 4 smoke JSON: [reports/smoke_phase4/phase4_multi_dataset_summary.json](reports/smoke_phase4/phase4_multi_dataset_summary.json)
- Colab GPU handoff: [docs/phase4_colab_gpu_handoff.md](docs/phase4_colab_gpu_handoff.md)
- Colab results template: [reports/colab_gpu_results.md](reports/colab_gpu_results.md)
- Final test run report: [reports/final_test_run.md](reports/final_test_run.md)
- Portfolio checklist: [docs/portfolio_readiness.md](docs/portfolio_readiness.md)
