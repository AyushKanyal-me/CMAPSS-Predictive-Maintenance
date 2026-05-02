# Portfolio Readiness Guide

This file is for shipping this project as a credible engineering portfolio item.

## 1) GitHub Publishing Checklist

- Add a clear repo description and topics (`predictive-maintenance`, `pytorch`, `mlflow`, `time-series`).
- Pin one representative issue and one representative notebook in the README.
- Keep generated artifacts out of Git unless they are intentionally versioned.
- Include reproducible commands for:
  - Safe local run.
  - Colab GPU run.
  - Test suite execution.
- Ensure `README.md` links to:
  - Implementation plan.
  - Modeling notebook.
  - Benchmark reports.
- Ensure Phase 4 full-run metrics and metadata are updated in `reports/`.
- Add 2-3 screenshots:
  - MLflow comparison view.
  - Data flow overview.
  - Final metrics table.

## 2) Authenticity Signals (Anti-"Vibecoded" Heuristics)

Recruiters and hiring managers usually trust projects that show constraint-driven engineering tradeoffs.

Keep these visible:

- Decision tradeoffs with rationale (not just model zoo results).
- Evidence of debugging iterations (for example: safe-mode profiling, fallback plans).
- Tests for leakage and boundary behavior.
- Concrete failure cases and limitations documented in plain language.

Avoid these patterns:

- Over-polished claims without reproducible commands.
- Huge architecture claims with no baseline comparisons.
- Missing metrics provenance.

## 3) Resume Bullet Options

Pick bullets that you can defend in interview follow-ups.

Option A (engineering + metrics)
- Built an end-to-end predictive maintenance pipeline on NASA C-MAPSS (FD001) with leakage-safe unit-level splits, MLflow experiment tracking, and sequence + tabular modeling; achieved RF validation RMSE 16.69 (vs LSTM 41.45 under CPU-safe constraints).

Option B (MLOps + reproducibility)
- Implemented reproducible modeling workflows with MLflow and phase-based artifacts (EDA, feature engineering, training), plus a phase 4 multi-dataset benchmark runner for FD001-FD004 with safe local defaults and Colab GPU handoff.

Option C (quality + reliability)
- Added unit tests for RUL construction, sequence boundary integrity, evaluation metrics, and multi-dataset dataset-resolution logic to reduce silent leakage/regression risk in time-series modeling.

## 4) Interview Prep Prompts

Be ready to answer:

- Why did Random Forest outperform the initial LSTM setup?
- What was done to prevent time leakage?
- How would you redesign the LSTM experiment on GPU?
- What changes are needed to operationalize this pipeline in production?
