# Retail Demand Forecast Engine

A production-grade retail demand forecasting platform built on the Kaggle M5 Forecasting hierarchical dataset (Walmart unit sales).

## Key Features

- **Multi-Model Forecasting Suite**: Baselines, Classical Statistical models (AutoARIMA, Croston), Gradient Boosted Trees (LightGBM/CatBoost), and Neural Sequence models.
- **Hierarchical Coherence**: MinT (Minimum Trace) and Top-Down reconciliation across all 12 Walmart hierarchy levels.
- **Uncertainty Quantification**: Split Conformal Prediction and Quantile intervals (P10, P50, P90).
- **MLOps & Monitoring**: MLflow experiment tracking & model registry, Population Stability Index (PSI) feature drift, and prediction drift monitoring.
- **Serving & UI**: High-performance FastAPI endpoint and interactive Streamlit analytics explorer.

## Quick Start

### 1. Installation
```bash
# Sync all dependencies including dev tools
uv sync --all-extras
```

### 2. Run Tests
```bash
uv run pytest
```

### 3. Start Services
```bash
# Start FastAPI server
uv run uvicorn api.main:app --reload --port 8000

# Start Streamlit UI
uv run streamlit run ui/Home.py --server.port 8501
```
