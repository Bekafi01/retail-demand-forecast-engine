"""Unit tests for evaluation metrics and temporal rolling window backtesting."""

import numpy as np
import pytest

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.backtest import RollingWindowSplitter
from src.evaluation.metrics import (
    WRMSSEEvaluator,
    compute_pinball_loss,
    compute_rmsse,
    compute_wape,
)


def test_compute_rmsse():
    """Verify RMSSE computation on known synthetic sequences."""
    y_train = np.array([10, 12, 11, 13, 12, 14], dtype=float)
    y_true = np.array([15, 14], dtype=float)
    y_pred_perfect = np.array([15, 14], dtype=float)
    y_pred_off = np.array([16, 15], dtype=float)

    assert compute_rmsse(y_train, y_true, y_pred_perfect) == 0.0
    assert compute_rmsse(y_train, y_true, y_pred_off) > 0.0


def test_compute_wape():
    """Verify WAPE computation."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 30.0])
    # abs errors: 1 + 1 + 0 = 2; total actual = 60; wape = 2/60 = 0.0333
    wape = compute_wape(y_true, y_pred)
    assert pytest.approx(wape, rel=1e-3) == 2.0 / 60.0


def test_compute_pinball_loss():
    """Verify Pinball / Quantile loss."""
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])  # errors: -2, +2
    # For median (q=0.5), pinball loss is 0.5 * MAE
    loss_50 = compute_pinball_loss(y_true, y_pred, quantile=0.5)
    assert pytest.approx(loss_50, rel=1e-3) == 1.0


def test_wrmsse_evaluator():
    """Verify WRMSSEEvaluator calculates scale, weights, and scored dictionary."""
    cal, prc, sal = generate_synthetic_m5_data(num_items=5, num_stores=2, num_days=60)
    sales_long = melt_sales_data(sal)
    df = merge_calendar_and_prices(sales_long, cal, prc)

    dates = df["date"].drop_duplicates().sort_values().values
    cutoff = dates[-28]

    train_df = df[df["date"] < cutoff]
    val_df = df[df["date"] >= cutoff]

    evaluator = WRMSSEEvaluator(train_df=train_df)
    assert len(evaluator.scales) == 10  # 5 items * 2 stores
    assert len(evaluator.weights) == 10
    assert pytest.approx(sum(evaluator.weights.values()), rel=1e-3) == 1.0

    # Perfect prediction test
    pred_df = val_df.copy()
    pred_df["y_pred"] = pred_df["sales"]

    scores = evaluator.score(val_df, pred_df)
    assert scores["wrmsse"] == 0.0
    assert scores["wape"] == 0.0


def test_rolling_window_splitter():
    """Verify temporal rolling window split generation has no forward leakage."""
    cal, prc, sal = generate_synthetic_m5_data(num_items=2, num_stores=2, num_days=120)
    sales_long = melt_sales_data(sal)
    df = merge_calendar_and_prices(sales_long, cal, prc)

    splitter = RollingWindowSplitter(horizon=28, n_splits=2, date_col="date")
    folds = list(splitter.split(df))

    assert len(folds) == 2
    for fold in folds:
        # Check no overlap between train and val
        assert fold.train_df["date"].max() <= fold.cutoff_date
        assert fold.val_df["date"].min() > fold.cutoff_date
        assert fold.val_df["date"].min() == fold.val_start_date
        assert fold.val_df["date"].max() == fold.val_end_date
        assert len(fold.val_df["date"].unique()) == 28
