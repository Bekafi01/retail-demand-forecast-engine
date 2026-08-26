"""Unit tests for FastAPI serving endpoints (/health, /predict, /drift/evaluate)."""

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint(client):
    """Verify /predict returns point forecasts and conformal bounds."""
    payload = {
        "items": [
            {
                "id": "FOODS_1_001_CA_1_evaluation",
                "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                "sell_price": 5.99,
                "date": "2016-05-23",
                "active_snap": 1,
                "price_discount_ratio": 0.15,
            }
        ],
        "horizon": 14,
        "reconcile": False,
        "include_conformal_intervals": True,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "LightGBM_Tweedie"
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert "y_pred" in pred
    assert "lower_90" in pred
    assert "upper_90" in pred
    assert pred["y_pred"] >= 0.0


def test_drift_evaluate_endpoint(client):
    """Verify /drift/evaluate endpoint calculates PSI from batch records."""
    payload = {
        "features": ["sell_price"],
        "current_data": [{"sell_price": 10.0 + i * 0.5} for i in range(20)],
    }

    response = client.post("/drift/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert "recommended_action" in data
    assert "feature_metrics" in data
