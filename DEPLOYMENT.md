# Deployment Guide

This project is ready for public GitHub publishing and a Render Docker deployment.

## 1) GitHub (public)

### Using GitHub CLI (recommended)

From the project directory:

```bash
git init

git add .

git commit -m "Finalize predictive maintenance project"

gh repo create cmapss-predictive-maintenance --public --source . --remote origin --push
```

If you are not logged in to GitHub CLI yet:

```bash
gh auth login
```

### Manual remote URL (if you prefer)

Create a public repo on GitHub, then:

```bash
git init

git add .

git commit -m "Finalize predictive maintenance project"

git remote add origin <your-repo-url>

git branch -M main

git push -u origin main
```

## 2) Render (Docker)

The repository includes a `Dockerfile` and `render.yaml` already configured.

### Model hosting (GitHub Release)

Upload `models/fd001_random_forest_baseline.joblib` as a Release asset and copy the direct URL.
Set `CMAPSS_BASELINE_MODEL_URL` to that URL in Render (Environment tab).

1. Push the repo to GitHub.
2. In Render, create a new Web Service and connect the repo.
3. Render should detect the Dockerfile automatically.
4. Keep `PORT=8000` and `healthCheckPath: /health` (already set in `render.yaml`).
5. Add `CMAPSS_BASELINE_MODEL_URL` pointing to the Release asset.
6. Deploy.

### Verify after deploy

- Health check: `https://<your-service>/health`
- Example request:

```bash
curl -X POST https://<your-service>/predict/baseline \
  -H "Content-Type: application/json" \
  --data-binary @docs/api_sample_request.json
```

## 3) Local smoke check

```bash
uvicorn src.api.app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict/baseline \
  -H "Content-Type: application/json" \
  --data-binary @docs/api_sample_request.json
```
