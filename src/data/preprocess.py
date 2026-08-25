"""Wide-to-long transformation, dataset merging, and parquet persistence."""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import polars as pl

from src.data.loader import reduce_mem_usage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def melt_sales_data(
    sales_df: Union[pd.DataFrame, pl.DataFrame],
    id_vars: Optional[List[str]] = None,
    day_prefix: str = "d_",
) -> pd.DataFrame:
    """Transform wide M5 sales data (with columns d_1, d_2, ...) into long format using fast Polars engine.

    Returns:
        DataFrame with columns [id_vars..., 'd', 'sales']
    """
    if id_vars is None:
        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    sales_pl = pl.from_pandas(sales_df) if isinstance(sales_df, pd.DataFrame) else sales_df

    present_id_vars = [col for col in id_vars if col in sales_pl.columns]
    day_cols = [c for c in sales_pl.columns if c.startswith(day_prefix)]

    logger.info(
        f"Melting sales data from {len(day_cols)} day columns across {len(sales_pl)} series..."
    )

    melted_pl = sales_pl.unpivot(
        index=present_id_vars,
        on=day_cols,
        variable_name="d",
        value_name="sales",
    )

    melted_pl = melted_pl.with_columns(pl.col("sales").cast(pl.Float32))
    return reduce_mem_usage(melted_pl.to_pandas())


def merge_calendar_and_prices(
    sales_long: Union[pd.DataFrame, pl.DataFrame],
    calendar_df: Union[pd.DataFrame, pl.DataFrame],
    prices_df: Union[pd.DataFrame, pl.DataFrame],
) -> pd.DataFrame:
    """Merge long sales records with calendar information and sell prices using Polars joins.

    Returns:
        Unified DataFrame enriched with calendar, events, SNAP, and pricing.
    """
    logger.info("Merging sales data with calendar...")

    sales_pl = pl.from_pandas(sales_long) if isinstance(sales_long, pd.DataFrame) else sales_long
    cal_pl = pl.from_pandas(calendar_df) if isinstance(calendar_df, pd.DataFrame) else calendar_df

    # Cast string types if needed for clean join
    sales_pl = sales_pl.with_columns(pl.col("d").cast(pl.Utf8))
    cal_pl = cal_pl.with_columns(pl.col("d").cast(pl.Utf8))

    merged_pl = sales_pl.join(cal_pl, on="d", how="left")

    if prices_df is not None and not (isinstance(prices_df, pd.DataFrame) and prices_df.empty):
        logger.info("Merging sell prices on store_id, item_id, and wm_yr_wk...")
        prices_pl = pl.from_pandas(prices_df) if isinstance(prices_df, pd.DataFrame) else prices_df

        # Ensure join keys have matching types
        merged_pl = merged_pl.with_columns(
            [
                pl.col("store_id").cast(pl.Utf8),
                pl.col("item_id").cast(pl.Utf8),
                pl.col("wm_yr_wk").cast(pl.Int32),
            ]
        )
        prices_pl = prices_pl.with_columns(
            [
                pl.col("store_id").cast(pl.Utf8),
                pl.col("item_id").cast(pl.Utf8),
                pl.col("wm_yr_wk").cast(pl.Int32),
            ]
        )

        merged_pl = merged_pl.join(prices_pl, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    merged_df = merged_pl.to_pandas()
    if "date" in merged_df.columns:
        merged_df["date"] = pd.to_datetime(merged_df["date"])

    return reduce_mem_usage(merged_df)


def build_processed_dataset(
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "data/processed/sales_long.parquet",
) -> pd.DataFrame:
    """Run full preprocessing pipeline and optionally save to compressed Parquet."""
    sales_long = melt_sales_data(sales_df)
    processed_df = merge_calendar_and_prices(sales_long, calendar_df, prices_df)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving processed dataset ({len(processed_df)} rows) to {out_p}...")
        processed_df.to_parquet(out_p, index=False, compression="snappy")

    return processed_df
