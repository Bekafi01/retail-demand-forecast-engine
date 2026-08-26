"""Pydantic schemas for the Demand Forecast Engine FastAPI service."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ForecastItem(BaseModel):
    id: str = Field(..., description="Unique series ID (e.g. FOODS_1_001_CA_1_evaluation)")
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str
    sell_price: float = Field(..., gt=0.0, description="Current selling price")
    date: str = Field(..., description="Forecast starting date (YYYY-MM-DD)")
    active_snap: Optional[int] = Field(0, description="1 if SNAP benefit is active, 0 otherwise")
    price_discount_ratio: Optional[float] = Field(
        0.0, description="Promotional discount ratio (0.0 to 1.0)"
    )


class ForecastRequest(BaseModel):
    items: List[ForecastItem] = Field(
        ..., min_length=1, description="List of series items to forecast"
    )
    horizon: int = Field(28, ge=1, le=90, description="Forecast horizon in days")
    reconcile: bool = Field(False, description="Apply hierarchical reconciliation if True")
    include_conformal_intervals: bool = Field(
        True, description="Include 80% and 90% conformal intervals"
    )


class ForecastPrediction(BaseModel):
    id: str
    date: str
    y_pred: float
    lower_90: Optional[float] = None
    upper_90: Optional[float] = None
    lower_80: Optional[float] = None
    upper_80: Optional[float] = None


class ForecastResponse(BaseModel):
    model_name: str
    horizon_days: int
    predictions: List[ForecastPrediction]
    is_hierarchically_reconciled: bool


class DriftRequest(BaseModel):
    features: Optional[List[str]] = None
    current_data: List[Dict[str, Any]] = Field(
        ..., min_length=5, description="Batch of current observation records"
    )


class DriftResponse(BaseModel):
    overall_status: str = Field(..., description="STABLE, MODERATE_DRIFT, or CRITICAL_DRIFT")
    recommended_action: str = Field(..., description="NO_ACTION, MONITOR, or TRIGGER_RETRAINING")
    num_features_checked: int
    num_critical_features: int
    feature_metrics: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_name: str
