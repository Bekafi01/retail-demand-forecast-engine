"""Unit tests for statistical and intermittent demand forecasting models."""

import pytest

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.backtest import run_backtest
from src.models.statistical import (
    AutoETSForecaster,
    AutoThetaForecaster,
    CrostonForecaster,
)


@pytest.fixture
def statistical_dataset():
    """Fixture providing processed synthetic sales dataset with intermittent items."""
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=4, num_stores=2, num_days=90, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    return merge_calendar_and_prices(sales_long, cal, prc)


def test_croston_sba(statistical_dataset):
    """Verify Croston SBA forecaster fits and predicts non-negative values on intermittent demand."""
    dates = statistical_dataset["date"].drop_duplicates().sort_values().values
    train_df = statistical_dataset[statistical_dataset["date"] < dates[-28]]
    val_df = statistical_dataset[statistical_dataset["date"] >= dates[-28]]

    model = CrostonForecaster(variant="sba")
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    assert "y_pred" in preds.columns
    assert len(preds) == len(val_df)
    assert (preds["y_pred"] >= 0.0).all()


def test_auto_theta_intervals(statistical_dataset):
    """Verify AutoTheta forecaster fits and produces calibrated prediction intervals."""
    dates = statistical_dataset["date"].drop_duplicates().sort_values().values
    train_df = statistical_dataset[statistical_dataset["date"] < dates[-28]]
    val_df = statistical_dataset[statistical_dataset["date"] >= dates[-28]]

    model = AutoThetaForecaster(season_length=7)
    model.fit(train_df)
    intervals = model.predict_intervals(val_df, horizon=28, quantiles=[0.1, 0.5, 0.9])

    assert "y_pred" in intervals.columns
    assert "q_10" in intervals.columns
    assert "q_50" in intervals.columns
    assert "q_90" in intervals.columns
    assert (intervals["q_10"] <= intervals["q_90"]).all()
    assert (intervals["y_pred"] >= 0.0).all()


def test_auto_ets(statistical_dataset):
    """Verify AutoETS forecaster execution."""
    dates = statistical_dataset["date"].drop_duplicates().sort_values().values
    train_df = statistical_dataset[statistical_dataset["date"] < dates[-28]]
    val_df = statistical_dataset[statistical_dataset["date"] >= dates[-28]]

    model = AutoETSForecaster(season_length=7)
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    assert "y_pred" in preds.columns
    assert len(preds) == len(val_df)
    assert (preds["y_pred"] >= 0.0).all()


def test_statistical_backtest_execution(statistical_dataset):
    """Verify backtesting runs cleanly with statistical models."""
    model = CrostonForecaster(variant="sba")
    agg_metrics, oof_df, fold_summaries = run_backtest(
        model=model,
        df=statistical_dataset,
        horizon=14,
        n_splits=2,
    )

    assert "mean_wrmsse" in agg_metrics
    assert "mean_wape" in agg_metrics
    assert len(fold_summaries) == 2
    assert len(oof_df) == len(statistical_dataset["id"].unique()) * 14 * 2
