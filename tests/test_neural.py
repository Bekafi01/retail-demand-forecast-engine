"""Unit tests for Deep Learning and MLP neural demand forecasters."""

import pandas as pd
import pytest

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.backtest import run_backtest
from src.features.pipeline import build_feature_table
from src.models.neural import MLPDemandForecaster


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Fixture providing prepared feature table from synthetic M5 data."""
    cal_df, prc_df, sal_df = generate_synthetic_m5_data(
        num_items=10, num_stores=2, num_days=180, random_seed=42
    )
    sales_long = melt_sales_data(sal_df)
    merged_df = merge_calendar_and_prices(sales_long, cal_df, prc_df)
    return build_feature_table(merged_df)


def test_mlp_fit_predict(sample_feature_data: pd.DataFrame) -> None:
    """Test MLPDemandForecaster fitting and 28-day inference."""
    dates = sorted(sample_feature_data["date"].unique())
    split_date = dates[int(len(dates) * 0.7)]
    train_df = sample_feature_data[sample_feature_data["date"] < split_date].copy()
    test_df = sample_feature_data[sample_feature_data["date"] >= split_date].copy()

    forecaster = MLPDemandForecaster(
        hidden_layer_sizes=(32, 16),
        max_iter=30,
        random_state=42,
    )
    forecaster.fit(train_df)

    assert forecaster.is_fitted
    assert forecaster.model is not None
    assert forecaster.scaler is not None

    preds = forecaster.predict(test_df, horizon=28)
    assert isinstance(preds, pd.DataFrame)
    assert "y_pred" in preds.columns
    assert len(preds) == len(test_df)
    assert (preds["y_pred"] >= 0.0).all()


def test_mlp_loss_curve(sample_feature_data: pd.DataFrame) -> None:
    """Test loss curve tracking during gradient descent."""
    forecaster = MLPDemandForecaster(
        hidden_layer_sizes=(32, 16),
        max_iter=40,
        random_state=42,
    )
    forecaster.fit(sample_feature_data)

    loss_curve = forecaster.loss_curve
    assert len(loss_curve) > 0
    assert isinstance(loss_curve, list)


def test_mlp_backtest(sample_feature_data: pd.DataFrame) -> None:
    """Test MLPDemandForecaster inside rolling backtesting engine."""
    forecaster = MLPDemandForecaster(
        hidden_layer_sizes=(32, 16),
        max_iter=20,
        random_state=42,
    )

    agg_metrics, oof_df, fold_summaries = run_backtest(
        model=forecaster,
        df=sample_feature_data,
        horizon=14,
        n_splits=2,
    )

    assert "mean_wrmsse" in agg_metrics
    assert "mean_wape" in agg_metrics
    assert agg_metrics["mean_wrmsse"] > 0
    assert len(fold_summaries) == 2
