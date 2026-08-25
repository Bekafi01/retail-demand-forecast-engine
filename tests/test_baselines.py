"""Unit tests for baseline demand forecasting models and backtest execution."""

import pytest

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.backtest import run_backtest
from src.models.baseline import (
    ExponentialSmoothingForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


@pytest.fixture
def sample_dataset():
    """Fixture providing processed synthetic sales dataset."""
    cal, prc, sal = generate_synthetic_m5_data(num_items=5, num_stores=2, num_days=100)
    sales_long = melt_sales_data(sal)
    return merge_calendar_and_prices(sales_long, cal, prc)


def test_naive_forecaster(sample_dataset):
    """Verify NaiveForecaster fits and predicts non-negative values."""
    dates = sample_dataset["date"].drop_duplicates().sort_values().values
    train_df = sample_dataset[sample_dataset["date"] < dates[-28]]
    val_df = sample_dataset[sample_dataset["date"] >= dates[-28]]

    model = NaiveForecaster()
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    assert "y_pred" in preds.columns
    assert len(preds) == len(val_df)
    assert (preds["y_pred"] >= 0.0).all()


def test_seasonal_naive_forecaster(sample_dataset):
    """Verify SeasonalNaiveForecaster fits and predicts with seasonality."""
    dates = sample_dataset["date"].drop_duplicates().sort_values().values
    train_df = sample_dataset[sample_dataset["date"] < dates[-28]]
    val_df = sample_dataset[sample_dataset["date"] >= dates[-28]]

    model = SeasonalNaiveForecaster(season_length=7)
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    assert "y_pred" in preds.columns
    assert len(preds) == len(val_df)
    assert (preds["y_pred"] >= 0.0).all()


def test_moving_average_forecaster(sample_dataset):
    """Verify MovingAverageForecaster fits and produces interval predictions."""
    dates = sample_dataset["date"].drop_duplicates().sort_values().values
    train_df = sample_dataset[sample_dataset["date"] < dates[-28]]
    val_df = sample_dataset[sample_dataset["date"] >= dates[-28]]

    model = MovingAverageForecaster(window=14)
    model.fit(train_df)
    intervals = model.predict_intervals(val_df, horizon=28, quantiles=[0.1, 0.5, 0.9])

    assert "y_pred" in intervals.columns
    assert "q_10" in intervals.columns
    assert "q_50" in intervals.columns
    assert "q_90" in intervals.columns
    assert (intervals["q_10"] <= intervals["q_90"]).all()


def test_backtest_execution(sample_dataset):
    """Verify end-to-end backtest runs over multiple folds and returns valid metrics."""
    model = ExponentialSmoothingForecaster(alpha=0.3)
    agg_metrics, oof_df, fold_summaries = run_backtest(
        model=model,
        df=sample_dataset,
        horizon=14,
        n_splits=2,
    )

    assert "mean_wrmsse" in agg_metrics
    assert "mean_wape" in agg_metrics
    assert len(fold_summaries) == 2
    assert len(oof_df) == len(sample_dataset["id"].unique()) * 14 * 2
