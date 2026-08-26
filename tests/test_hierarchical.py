"""Unit tests for hierarchical aggregation, MinT reconciliation, and conformal prediction."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.conformal import ConformalCalibrator
from src.models.hierarchical import HierarchicalReconciler, aggregate_hierarchy


@pytest.fixture
def hierarchical_dataset():
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=4, num_stores=2, num_days=60, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    return merge_calendar_and_prices(sales_long, cal, prc)


def test_aggregate_hierarchy_and_summing_matrix(hierarchical_dataset):
    """Verify S matrix construction and exact mathematical sum matching."""
    hier_df, bottom_meta, S, all_nodes, bottom_nodes = aggregate_hierarchy(
        hierarchical_dataset,
        hierarchy_levels=[["total"], ["state_id"], ["id"]],
    )

    assert len(all_nodes) > len(bottom_nodes)
    assert S.shape[0] == len(all_nodes)
    assert S.shape[1] == len(bottom_nodes)

    # Check total sum on a single date
    sample_date = hierarchical_dataset["date"].iloc[0]
    total_sales = hier_df[(hier_df["date"] == sample_date) & (hier_df["node_id"] == "Total")][
        "sales"
    ].values[0]
    bottom_sum = hier_df[(hier_df["date"] == sample_date) & (hier_df["level"] == "id")][
        "sales"
    ].sum()

    assert np.isclose(total_sales, bottom_sum)


def test_hierarchical_reconciliation_coherence(hierarchical_dataset):
    """Verify MinT and Bottom-Up reconciliation guarantee mathematical coherence."""
    hier_df, bottom_meta, S, all_nodes, bottom_nodes = aggregate_hierarchy(
        hierarchical_dataset,
        hierarchy_levels=[["total"], ["state_id"], ["id"]],
    )

    # Create un-reconciled dummy base forecasts (with intentional incoherence)
    dates = hier_df["date"].drop_duplicates().values[:5]
    base_preds = []
    for d in dates:
        for node in all_nodes:
            base_preds.append(
                {
                    "date": d,
                    "node_id": node,
                    "y_pred": np.random.uniform(5.0, 15.0),
                }
            )
    base_df = pd.DataFrame(base_preds)

    for method in ["bottom_up", "ols", "wls_struct", "mint_shrink"]:
        reconciler = HierarchicalReconciler(method=method)
        reconciler.fit(S, all_nodes, bottom_nodes)
        rec_df = reconciler.reconcile(base_df)

        assert "y_reconciled" in rec_df.columns
        assert (rec_df["y_reconciled"] >= 0.0).all()
        assert reconciler.check_coherence(rec_df)


def test_conformal_calibrator():
    """Verify Split Conformal Prediction achieves valid empirical coverage."""
    np.random.seed(42)
    n = 200
    y_true = np.random.normal(10.0, 2.0, n)
    y_pred = y_true + np.random.normal(0.0, 1.0, n)

    # Split into calibration and test
    cal_true, test_true = y_true[:100], y_true[100:]
    cal_pred, test_pred = y_pred[:100], y_pred[100:]

    calibrator = ConformalCalibrator()
    calibrator.fit(cal_true, cal_pred)

    test_df = pd.DataFrame({"y_pred": test_pred})
    intervals_df = calibrator.predict_intervals(test_df, alphas=[0.1])

    assert "lower_90" in intervals_df.columns
    assert "upper_90" in intervals_df.columns
    assert (intervals_df["lower_90"] <= intervals_df["upper_90"]).all()

    eval_res = ConformalCalibrator.evaluate_coverage(
        test_true,
        intervals_df["lower_90"],
        intervals_df["upper_90"],
        alpha=0.1,
    )
    # Check that empirical coverage is close to target (>= 80%)
    assert eval_res["empirical_coverage"] >= 0.80
