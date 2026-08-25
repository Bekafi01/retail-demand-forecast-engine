"""Tests for data loading, synthetic data generation, memory optimization, and preprocessing."""

import pandas as pd

from src.data.loader import generate_synthetic_m5_data, reduce_mem_usage
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices


def test_synthetic_data_generation():
    """Verify synthetic M5 data generator produces valid schemas and non-empty DataFrames."""
    calendar_df, prices_df, sales_df = generate_synthetic_m5_data(
        num_items=10,
        num_stores=2,
        num_days=30,
        random_seed=42,
    )

    assert not calendar_df.empty
    assert not prices_df.empty
    assert not sales_df.empty

    # Check calendar schema
    expected_cal_cols = {
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "d",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    }
    assert expected_cal_cols.issubset(set(calendar_df.columns))
    assert len(calendar_df) == 30

    # Check sales wide schema
    assert len(sales_df) == 20  # 10 items * 2 stores
    assert "d_1" in sales_df.columns
    assert "d_30" in sales_df.columns
    assert "id" in sales_df.columns
    assert "item_id" in sales_df.columns
    assert "store_id" in sales_df.columns


def test_reduce_mem_usage():
    """Verify memory downcasting converts types appropriately."""
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3, 100] * 25,
            "float_col": [1.5, 2.5, 3.5, 4.5] * 25,
            "cat_col": ["A", "B", "A", "B"] * 25,
        }
    )

    downcasted = reduce_mem_usage(df)
    assert downcasted["int_col"].dtype == "int8"
    assert downcasted["float_col"].dtype == "float32"
    assert str(downcasted["cat_col"].dtype) == "category"


def test_melting_and_merging():
    """Verify wide-to-long melting and calendar/price joining."""
    calendar_df, prices_df, sales_df = generate_synthetic_m5_data(
        num_items=5,
        num_stores=2,
        num_days=14,
        random_seed=42,
    )

    sales_long = melt_sales_data(sales_df)
    assert len(sales_long) == 5 * 2 * 14  # 140 rows
    assert "sales" in sales_long.columns
    assert "d" in sales_long.columns

    merged = merge_calendar_and_prices(sales_long, calendar_df, prices_df)
    assert len(merged) == 140
    assert "date" in merged.columns
    assert "sell_price" in merged.columns
    assert pd.api.types.is_datetime64_any_dtype(merged["date"])
