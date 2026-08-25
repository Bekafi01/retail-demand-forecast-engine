"""Unit tests for LightGBM and CatBoost GBDT forecasters."""

import pytest

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.backtest import run_backtest
from src.features.pipeline import build_feature_table
from src.models.gbm import CatBoostForecaster, LightGBMForecaster


@pytest.fixture
def gbm_dataset():
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=4, num_stores=2, num_days=120, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    merged = merge_calendar_and_prices(sales_long, cal, prc)
    return build_feature_table(merged)


def test_lightgbm_fit_predict(gbm_dataset):
    """Verify LightGBM model fits and produces non-negative predictions and feature importances."""
    dates = gbm_dataset["date"].drop_duplicates().sort_values().values
    train_df = gbm_dataset[gbm_dataset["date"] < dates[-28]]
    val_df = gbm_dataset[gbm_dataset["date"] >= dates[-28]]

    model = LightGBMForecaster(n_estimators=30, learning_rate=0.1)
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    assert "y_pred" in preds.columns
    assert len(preds) == len(val_df)
    assert (preds["y_pred"] >= 0.0).all()

    # Check feature importances
    imp_df = model.get_feature_importances()
    assert not imp_df.empty
    assert "feature" in imp_df.columns
    assert "importance" in imp_df.columns


def test_catboost_fit_predict(gbm_dataset):
    """Verify CatBoost model fits and predicts non-negative values."""
    dates = gbm_dataset["date"].drop_duplicates().sort_values().values
    train_df = gbm_dataset[gbm_dataset["date"] < dates[-28]]
    val_df = gbm_dataset[gbm_dataset["date"] >= dates[-28]]

    model = CatBoostForecaster(iterations=20, learning_rate=0.1)
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    assert "y_pred" in preds.columns
    assert len(preds) == len(val_df)
    assert (preds["y_pred"] >= 0.0).all()


def test_lightgbm_backtest_execution(gbm_dataset):
    """Verify LightGBM completes temporal rolling backtesting."""
    model = LightGBMForecaster(n_estimators=30, learning_rate=0.1)
    agg_metrics, oof_df, fold_summaries = run_backtest(
        model=model,
        df=gbm_dataset,
        horizon=14,
        n_splits=2,
    )

    assert "mean_wrmsse" in agg_metrics
    assert "mean_wape" in agg_metrics
    assert len(fold_summaries) == 2
