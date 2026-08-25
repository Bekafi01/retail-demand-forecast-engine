"""Unit tests for feature engineering pipeline and target leakage prevention."""

import pytest

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.features.calendar import extract_calendar_features
from src.features.lags import extract_lag_and_rolling_features
from src.features.pipeline import build_feature_table, get_feature_column_names
from src.features.price import extract_price_features


@pytest.fixture
def raw_merged_dataset():
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=5, num_stores=2, num_days=90, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    return merge_calendar_and_prices(sales_long, cal, prc)


def test_calendar_features(raw_merged_dataset):
    """Verify temporal, cyclical, and SNAP feature extraction."""
    featured = extract_calendar_features(raw_merged_dataset)

    expected_cols = {
        "dayofweek",
        "month",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "active_snap",
    }
    assert expected_cols.issubset(set(featured.columns))
    assert featured["dow_sin"].between(-1.0, 1.0).all()
    assert featured["dow_cos"].between(-1.0, 1.0).all()
    assert set(featured["is_weekend"].unique()).issubset({0, 1})


def test_price_features(raw_merged_dataset):
    """Verify price discount and momentum calculations."""
    featured = extract_price_features(raw_merged_dataset)

    expected_cols = {"price_max", "price_min", "price_discount_ratio", "price_momentum_w"}
    assert expected_cols.issubset(set(featured.columns))
    assert (featured["price_discount_ratio"] >= 0.0).all()


def test_leak_free_lags(raw_merged_dataset):
    """Verify lag features are strictly shifted by at least 28 days to prevent target leakage."""
    featured = extract_lag_and_rolling_features(
        raw_merged_dataset,
        lags=[28, 35],
        rolling_windows=[7, 28],
        base_shift=28,
    )

    assert "sales_lag_28" in featured.columns
    assert "rolling_mean_7_lag_28" in featured.columns

    # Verify that for the first 28 days of each series, sales_lag_28 is NaN
    for sid, grp in featured.groupby("id", observed=False):
        grp_sorted = grp.sort_values("date")
        assert grp_sorted["sales_lag_28"].iloc[:28].isna().all()
        # Verify 29th day equals the 1st day's sales
        assert grp_sorted["sales_lag_28"].iloc[28] == grp_sorted["sales"].iloc[0]


def test_build_feature_table(raw_merged_dataset):
    """Verify end-to-end feature table pipeline."""
    feat_df = build_feature_table(raw_merged_dataset)
    feature_cols = get_feature_column_names(feat_df)

    assert len(feature_cols) > 15
    assert "sales_lag_28" in feature_cols
    assert "dow_sin" in feature_cols
    assert "price_discount_ratio" in feature_cols
    assert "sales" not in feature_cols
    assert "date" not in feature_cols
