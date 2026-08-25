"""Price elasticity, promotional discount, and price momentum feature engineering."""

from typing import Union

import pandas as pd
import polars as pl

from src.data.loader import reduce_mem_usage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_price_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    price_col: str = "sell_price",
    item_col: str = "item_id",
    store_col: str = "store_id",
    dept_col: str = "dept_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Extract promotional discount depth, historical price momentum, and relative category price."""
    df_pl = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df

    if price_col not in df_pl.columns or df_pl[price_col].null_count() == len(df_pl):
        logger.warning(
            f"Price column '{price_col}' not available or empty. Generating default price features."
        )
        return reduce_mem_usage(
            df_pl.with_columns(
                [
                    pl.lit(1.0).cast(pl.Float32).alias("sell_price"),
                    pl.lit(0.0).cast(pl.Float32).alias("price_discount_ratio"),
                    pl.lit(0.0).cast(pl.Float32).alias("price_momentum_w"),
                    pl.lit(1.0).cast(pl.Float32).alias("price_relative_dept"),
                ]
            ).to_pandas()
        )

    logger.info("Extracting price elasticity, discount depth, and momentum features...")
    df_pl = df_pl.sort([store_col, item_col, date_col])

    # 1. Historical price extremes per (store_id, item_id)
    store_item_keys = [store_col, item_col]
    df_pl = df_pl.with_columns(
        [
            pl.col(price_col).max().over(store_item_keys).alias("price_max"),
            pl.col(price_col).min().over(store_item_keys).alias("price_min"),
            pl.col(price_col).mean().over(store_item_keys).alias("price_mean"),
            pl.col(price_col).std().over(store_item_keys).alias("price_std"),
        ]
    )

    # 2. Discount depth ratio (how deeply discounted is the item relative to historical peak)
    df_pl = df_pl.with_columns(
        [
            ((pl.col("price_max") - pl.col(price_col)) / (pl.col("price_max") + 1e-4))
            .cast(pl.Float32)
            .alias("price_discount_ratio"),
            (pl.col(price_col) / (pl.col("price_mean") + 1e-4))
            .cast(pl.Float32)
            .alias("price_to_mean_ratio"),
        ]
    )

    # 3. Weekly & monthly price momentum
    df_pl = df_pl.with_columns(
        [
            (
                (pl.col(price_col) - pl.col(price_col).shift(7).over(store_item_keys))
                / (pl.col(price_col).shift(7).over(store_item_keys) + 1e-4)
            )
            .cast(pl.Float32)
            .fill_null(0.0)
            .alias("price_momentum_w"),
            (
                (pl.col(price_col) - pl.col(price_col).shift(28).over(store_item_keys))
                / (pl.col(price_col).shift(28).over(store_item_keys) + 1e-4)
            )
            .cast(pl.Float32)
            .fill_null(0.0)
            .alias("price_momentum_m"),
        ]
    )

    # 4. Relative price within department
    if dept_col in df_pl.columns:
        df_pl = df_pl.with_columns(
            (
                pl.col(price_col)
                / (pl.col(price_col).mean().over([store_col, dept_col, date_col]) + 1e-4)
            )
            .cast(pl.Float32)
            .alias("price_relative_dept")
        )

    return reduce_mem_usage(df_pl.to_pandas())
