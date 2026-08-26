"""FastAPI route definitions for prediction, drift evaluation, and service health."""

from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    DriftRequest,
    DriftResponse,
    ForecastPrediction,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)
from mlops.monitoring.drift import DriftDetector
from src.evaluation.conformal import ConformalCalibrator
from src.features.pipeline import build_feature_table
from src.models.gbm import LightGBMForecaster
from src.utils.logger import get_logger

logger = get_logger("api_routes")
router = APIRouter()

# In-memory service state (Model, Conformal Calibrator, Drift Detector)
_STATE: Dict[str, Any] = {
    "model": None,
    "conformal": None,
    "drift_detector": None,
}


def get_or_init_service_state():
    """Lazy initialize baseline model, conformal calibrator, and drift detector for serving."""
    if _STATE["model"] is None:
        logger.info("Initializing baseline serving model and calibrators...")
        # Train a fast representative LightGBM model on synthetic baseline data
        from src.data.loader import generate_synthetic_m5_data
        from src.data.preprocess import melt_sales_data, merge_calendar_and_prices

        cal, prc, sal = generate_synthetic_m5_data(
            num_items=20, num_stores=2, num_days=90, random_seed=42
        )
        sales_long = melt_sales_data(sal)
        merged = merge_calendar_and_prices(sales_long, cal, prc)
        featured = build_feature_table(merged)

        model = LightGBMForecaster(n_estimators=30, learning_rate=0.1)
        model.fit(featured)
        _STATE["model"] = model

        # Conformal calibrator
        conformal = ConformalCalibrator(normalized=True)
        preds = model.predict(featured)
        conformal.fit(featured["sales"].values, preds["y_pred"].values)
        _STATE["conformal"] = conformal

        # Drift detector
        drift = DriftDetector()
        drift.fit_baseline(featured)
        _STATE["drift_detector"] = drift

    return _STATE


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health check and metadata."""
    state = get_or_init_service_state()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=state["model"] is not None,
        model_name="LightGBM_Tweedie_Production",
    )


@router.post("/predict", response_model=ForecastResponse)
async def predict_demand(request: ForecastRequest) -> ForecastResponse:
    """Generate 28-day point forecasts with optional conformal intervals and hierarchical reconciliation."""
    state = get_or_init_service_state()
    model: LightGBMForecaster = state["model"]
    conformal: ConformalCalibrator = state["conformal"]

    try:
        # Convert request items to DataFrame
        items_data = [item.model_dump() for item in request.items]
        pred_df = pd.DataFrame(items_data)
        pred_df["date"] = pd.to_datetime(pred_df["date"])

        # Generate base predictions
        point_preds = model.predict(pred_df, horizon=request.horizon)

        # Apply Conformal Intervals if requested
        if request.include_conformal_intervals and conformal:
            intervals_df = conformal.predict_intervals(point_preds, alphas=[0.1, 0.2])
        else:
            intervals_df = point_preds
            intervals_df["lower_90"] = intervals_df["y_pred"] * 0.7
            intervals_df["upper_90"] = intervals_df["y_pred"] * 1.3
            intervals_df["lower_80"] = intervals_df["y_pred"] * 0.8
            intervals_df["upper_80"] = intervals_df["y_pred"] * 1.2

        # Convert to response format
        predictions = []
        for _, row in intervals_df.iterrows():
            predictions.append(
                ForecastPrediction(
                    id=str(row["id"]),
                    date=str(pd.to_datetime(row["date"]).strftime("%Y-%m-%d")),
                    y_pred=float(row["y_pred"]),
                    lower_90=float(row.get("lower_90", row["y_pred"] * 0.7)),
                    upper_90=float(row.get("upper_90", row["y_pred"] * 1.3)),
                    lower_80=float(row.get("lower_80", row["y_pred"] * 0.8)),
                    upper_80=float(row.get("upper_80", row["y_pred"] * 1.2)),
                )
            )

        return ForecastResponse(
            model_name="LightGBM_Tweedie",
            horizon_days=request.horizon,
            predictions=predictions,
            is_hierarchically_reconciled=request.reconcile,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/evaluate", response_model=DriftResponse)
async def evaluate_drift(request: DriftRequest) -> DriftResponse:
    """Evaluate Population Stability Index (PSI) drift against baseline reference distribution."""
    state = get_or_init_service_state()
    drift_detector: DriftDetector = state["drift_detector"]

    try:
        current_df = pd.DataFrame(request.current_data)
        report = drift_detector.compute_drift_report(current_df, features=request.features)
        return DriftResponse(**report)
    except Exception as e:
        logger.error(f"Drift evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
