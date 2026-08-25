"""High-performance data loading, memory optimization, and synthetic M5 dataset generation."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns to the smallest possible type and convert object columns to category."""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != "object" and not isinstance(col_type, pd.CategoricalDtype):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == "float":
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == "object":
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(
            f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB "
            f"({(start_mem - end_mem) / start_mem * 100:.1f}% reduction)"
        )
    return df


def load_raw_data(
    data_dir: Union[str, Path] = "data/raw",
    nrows: Optional[int] = None,
    sales_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw calendar, sell_prices, and sales_train datasets using fast Polars reader with Pandas conversion.

    Returns:
        Tuple[calendar_df, prices_df, sales_df]
    """
    data_path = Path(data_dir)

    calendar_path = data_path / "calendar.csv"
    prices_path = data_path / "sell_prices.csv"
    sales_path = data_path / (sales_file or "sales_train_evaluation.csv")

    if not sales_path.exists():
        fallback = data_path / "sales_train_validation.csv"
        if fallback.exists():
            sales_path = fallback

    if not calendar_path.exists() or not sales_path.exists():
        raise FileNotFoundError(
            f"Required M5 raw data files not found in {data_path}. "
            f"Ensure calendar.csv, sell_prices.csv, and sales_train_evaluation.csv are placed in {data_path}, "
            f"or use generate_synthetic_m5_data() for testing."
        )

    logger.info(f"Loading calendar from {calendar_path}...")
    calendar_df = pl.read_csv(calendar_path).to_pandas()
    calendar_df = reduce_mem_usage(calendar_df)

    logger.info(f"Loading sales data from {sales_path} (nrows={nrows})...")
    if nrows:
        sales_pl = pl.read_csv(sales_path, n_rows=nrows)
    else:
        sales_pl = pl.read_csv(sales_path)
    sales_df = reduce_mem_usage(sales_pl.to_pandas())

    logger.info(f"Loading sell prices from {prices_path}...")
    if prices_path.exists():
        if nrows:
            # Filter prices lazily to only matching items and stores
            unique_items = sales_pl["item_id"].unique().to_list()
            unique_stores = sales_pl["store_id"].unique().to_list()
            prices_pl = (
                pl.scan_csv(prices_path)
                .filter(
                    pl.col("store_id").is_in(unique_stores) & pl.col("item_id").is_in(unique_items)
                )
                .collect()
            )
            prices_df = prices_pl.to_pandas()
        else:
            prices_df = pl.read_csv(prices_path).to_pandas()
        prices_df = reduce_mem_usage(prices_df)
    else:
        prices_df = pd.DataFrame()

    return calendar_df, prices_df, sales_df


def generate_synthetic_m5_data(
    num_items: int = 50,
    num_stores: int = 4,
    num_days: int = 365,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic M5 hierarchical data for offline testing and CI.

    Hierarchies:
    - 2 States (CA, TX)
    - 2 Stores per State (CA_1, CA_2, TX_1, TX_2)
    - 2 Categories (HOBBIES, FOODS)
    - 2 Departments per Category (HOBBIES_1, HOBBIES_2, FOODS_1, FOODS_2)
    - Items mapped to departments

    Returns:
        Tuple[calendar_df, prices_df, sales_df]
    """
    np.random.seed(random_seed)

    # 1. Generate Calendar
    start_date = pd.Timestamp("2015-01-01")
    dates = [start_date + pd.Timedelta(days=i) for i in range(num_days)]

    calendar_records = []
    for i, dt in enumerate(dates):
        d_str = f"d_{i + 1}"
        wm_yr_wk = 11501 + (i // 7)
        wday = dt.dayofweek + 1
        weekday = dt.day_name()
        month = dt.month
        year = dt.year

        # Exogenous events and SNAP
        event_name_1 = np.random.choice(
            ["SuperBowl", "ValentinesDay", "Easter", "MemorialDay", None],
            p=[0.02, 0.02, 0.02, 0.02, 0.92],
        )
        event_type_1 = (
            "Sporting" if event_name_1 == "SuperBowl" else ("Cultural" if event_name_1 else None)
        )
        snap_CA = 1 if dt.day <= 10 and np.random.rand() > 0.1 else 0
        snap_TX = 1 if dt.day in [1, 3, 5, 6, 7, 9, 11] and np.random.rand() > 0.1 else 0
        snap_WI = 1 if dt.day in [2, 4, 6, 8, 10] and np.random.rand() > 0.1 else 0

        calendar_records.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "wm_yr_wk": wm_yr_wk,
                "weekday": weekday,
                "wday": wday,
                "month": month,
                "year": year,
                "d": d_str,
                "event_name_1": event_name_1,
                "event_type_1": event_type_1,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": snap_CA,
                "snap_TX": snap_TX,
                "snap_WI": snap_WI,
            }
        )
    calendar_df = pd.DataFrame(calendar_records)

    # 2. Generate Store & Item Hierarchy
    states = ["CA", "TX"]
    stores = [f"{st}_{i}" for st in states for i in range(1, (num_stores // len(states)) + 1)]
    categories = ["HOBBIES", "FOODS"]
    departments = {
        "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
        "FOODS": ["FOODS_1", "FOODS_2"],
    }

    items = []
    for item_idx in range(1, num_items + 1):
        cat = categories[item_idx % len(categories)]
        dept = departments[cat][item_idx % len(departments[cat])]
        item_id = f"{dept}_{item_idx:03d}"
        items.append(
            {
                "item_id": item_id,
                "dept_id": dept,
                "cat_id": cat,
            }
        )
    items_df = pd.DataFrame(items)

    # Cross join items and stores
    series_list = []
    prices_list = []
    unique_wm_yr_wks = calendar_df["wm_yr_wk"].unique()

    for store in stores:
        state = store.split("_")[0]
        for _, item in items_df.iterrows():
            series_id = f"{item['item_id']}_{store}_evaluation"

            # Base demand parameters
            base_rate = (
                np.random.uniform(0.5, 4.0)
                if item["cat_id"] == "FOODS"
                else np.random.uniform(0.1, 1.5)
            )
            intermittency_prob = 0.2 if item["cat_id"] == "FOODS" else 0.5

            # Generate sales sequence across days
            daily_sales = []
            for i, dt in enumerate(dates):
                # Day of week effect
                dow_mult = 1.3 if dt.dayofweek in [5, 6] else 1.0
                # SNAP effect
                snap_active = (state == "CA" and calendar_records[i]["snap_CA"] == 1) or (
                    state == "TX" and calendar_records[i]["snap_TX"] == 1
                )
                snap_mult = 1.25 if (snap_active and item["cat_id"] == "FOODS") else 1.0
                # Trend / seasonality
                month_mult = 1.0 + 0.15 * np.sin(2 * np.pi * dt.month / 12)

                rate = base_rate * dow_mult * snap_mult * month_mult
                if np.random.rand() < intermittency_prob:
                    sale = 0
                else:
                    sale = np.random.poisson(rate)
                daily_sales.append(sale)

            series_row = {
                "id": series_id,
                "item_id": item["item_id"],
                "dept_id": item["dept_id"],
                "cat_id": item["cat_id"],
                "store_id": store,
                "state_id": state,
            }
            for d_idx, s in enumerate(daily_sales):
                series_row[f"d_{d_idx + 1}"] = s
            series_list.append(series_row)

            # Generate prices per store/item
            base_price = np.random.uniform(1.5, 12.0)
            for wk in unique_wm_yr_wks:
                # Occasional discount
                discount = 0.85 if np.random.rand() < 0.15 else 1.0
                prices_list.append(
                    {
                        "store_id": store,
                        "item_id": item["item_id"],
                        "wm_yr_wk": wk,
                        "sell_price": round(base_price * discount, 2),
                    }
                )

    sales_df = pd.DataFrame(series_list)
    prices_df = pd.DataFrame(prices_list)

    return (
        reduce_mem_usage(calendar_df),
        reduce_mem_usage(prices_df),
        reduce_mem_usage(sales_df),
    )
