# Retail Demand Forecast Engine

A production-grade, end-to-end retail demand forecasting platform built on the Kaggle M5 Forecasting dataset (Walmart hierarchical unit sales data).

---

## 🌟 Key Features

- **Multi-Model Forecasting Suite**:
  - *Baselines*: Naive, Seasonal Naive (7-day), Moving Average (28-day), Exponential Smoothing.
  - *Statistical*: Croston (Classic & SBA for intermittent zero-inflated demand), AutoTheta, AutoETS, AutoARIMA.
  - *Gradient Boosted Trees*: LightGBM and CatBoost with Compound Poisson-Gamma **Tweedie Loss** ($p = 1.15$).
  - *Deep Learning*: Deep Multi-Layer Perceptrons (**MLP**) with latent embeddings, standard scaling, and Adam optimization.
- **Leak-Free Feature Store**:
  - Polars-accelerated $\ge 28$-day shifted lags, rolling window aggregations (mean, std, min, max), promotional discount depth, price momentum, cyclical calendar encodings, and state-level SNAP benefit flags (`snap_CA`, `snap_TX`, `snap_WI`).
- **Hierarchical Reconciliation**:
  - Summing Matrix ($S$) construction and **MinT (Minimum Trace)** reconciliation with Structural WLS and Shrinkage covariance weighting across all Walmart organizational levels.
- **Uncertainty Quantification**:
  - **Split Conformal Prediction (SCP)** generating distribution-free prediction intervals (80% and 90% confidence bands) with finite-sample coverage guarantees.
- **MLOps & Drift Monitoring**:
  - **MLflow Tracking & Model Registry**: Automatic logging of hyperparameters, rolling backtest validation metrics (WRMSSE, WAPE, RMSE), and Champion vs Challenger promotion.
  - **Population Stability Index (PSI)** & KS-test data drift detection with automated retraining triggers.
- **Production Serving & Analytics UI**:
  - High-performance **FastAPI** async service with Pydantic validation for point forecasts, conformal intervals, and drift evaluation.
  - Interactive **Streamlit** dashboard featuring Forecast Explorer, What-If Scenario Pricing Simulator, and MLOps Drift Monitor.
- **Enterprise CI/CD & Containerization**:
  - Multi-target `Dockerfile` using `uv` frozen dependencies, `docker-compose.yml` for unified FastAPI + Streamlit + MLflow deployment, and GitHub Actions workflows for automated testing and scheduled weekly retraining.

---

## 🏗️ Project Architecture

```
retail-demand-forecast-engine/
├── .github/workflows/
│   ├── ci.yml                             # Automated linting & pytest test suite
│   └── retrain.yml                        # Scheduled / manual model retraining workflow
├── configs/
│   └── model_config.yaml                  # Central configuration
├── data/
│   ├── raw/                               # Raw Kaggle M5 CSVs (calendar, prices, sales)
│   └── processed/                         # Memory-reduced parquet feature stores
├── src/
│   ├── data/                              # Ingestion, downcasting, and synthetic generator
│   ├── features/                          # Calendar, SNAP, price momentum, shifted lags
│   ├── models/                            # Baselines, Croston, AutoTheta, LightGBM, Neural MLP, MinT
│   ├── evaluation/                        # WRMSSE, WAPE, Rolling Backtesting, Conformal Calibrator
│   └── utils/                             # Logger and Pydantic configuration loader
├── mlops/
│   ├── tracking/                          # MLflow experiment tracking wrapper
│   ├── registry/                          # Model registry & Champion vs Challenger promotion
│   ├── monitoring/                        # Population Stability Index (PSI) drift detector
│   └── pipeline/                          # CLI training pipeline orchestrator
├── api/
│   ├── app.py                             # FastAPI service
│   ├── routes.py                          # /predict, /health, and /drift/evaluate routes
│   └── schemas.py                         # Pydantic v2 schemas
├── ui/
│   ├── app.py                             # Streamlit main dashboard
│   └── pages/                             # Forecast Explorer, Simulator, Drift Monitor
├── reports/
│   ├── figures/                           # Benchmark plots (master leaderboard, pareto frontier)
│   └── final_report.md                    # Executive benchmark report & supply chain impact
├── notebooks/                             # 8-part exploratory and benchmarking series
├── tests/                                 # 35 unit tests across all engine modules
├── Dockerfile                             # Multi-stage uv container
├── docker-compose.yml                     # Unified multi-container orchestration
└── pyproject.toml                         # uv-managed dependencies and tooling
```

---

## 🚀 Quick Start

### 1. Installation
Ensure [uv](https://docs.astral.sh/uv/) is installed:
```bash
uv sync --all-extras
```

### 2. Run the Full Test Suite
```bash
uv run pytest -v
```

### 3. Start the Services Locally
```bash
# Start FastAPI Serving Microservice
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Launch Streamlit Analytics Dashboard
uv run streamlit run ui/app.py

# Execute Retraining Pipeline
uv run python mlops/pipeline/train_pipeline.py --config configs/model_config.yaml
```

### 4. Docker Deployment
```bash
# Build and launch all 3 services (FastAPI, Streamlit, MLflow Server)
docker compose up --build
```

---

## 📓 Notebook Series

1. **[01_eda.ipynb](notebooks/01_eda.ipynb)**: Hierarchy structure, intermittency, SNAP patterns, price elasticity, and stationarity tests.
2. **[02_baselines_backtest.ipynb](notebooks/02_baselines_backtest.ipynb)**: Rolling temporal cross-validation and official WRMSSE/WAPE benchmark.
3. **[03_statistical_models.ipynb](notebooks/03_statistical_models.ipynb)**: Croston SBA, AutoTheta, AutoETS, and intermittent demand slicing.
4. **[04_ml_models.ipynb](notebooks/04_ml_models.ipynb)**: LightGBM and CatBoost with Tweedie loss, Gain feature importance, and multi-step forecasting.
5. **[05_hierarchical_conformal.ipynb](notebooks/05_hierarchical_conformal.ipynb)**: Summing Matrix $S$, MinT hierarchical reconciliation, and Split Conformal Prediction intervals.
6. **[06_mlops_monitoring.ipynb](notebooks/06_mlops_monitoring.ipynb)**: MLflow experiment tracking, Champion model registration, and PSI data drift simulation.
7. **[07_model_benchmark_report.ipynb](notebooks/07_model_benchmark_report.ipynb)**: Master comparative benchmark matrix, Pareto latency-accuracy frontier, and statistical significance testing.
8. **[08_deep_learning.ipynb](notebooks/08_deep_learning.ipynb)**: Deep MLP neural architecture, Adam loss curves, and head-to-head backtesting vs LightGBM.
