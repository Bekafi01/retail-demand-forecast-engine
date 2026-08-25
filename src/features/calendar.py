"""Calendar, event, SNAP benefits, and cyclical time feature engineering."""

from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from src.data.loader import reduce_mem_usage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_calendar_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    date_col: str = "date",
    state_col: str = "state_id",
) -> pd.DataFrame:
    """Extract temporal, cyclical, and SNAP benefit features from dataset.

    Features created:
    - Calendar parts: dayofweek, day, month, year, week, is_weekend
    - Cyclical encodings: sin/cos for day of week (period 7), month (period 12), day of month (period 31)
    - State-specific active SNAP indicator: active_snap
    - Event indicators: is_event_1, is_event_2
    """
    logger.info("Extracting calendar and cyclical temporal features...")
    df_pl = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df

    # Ensure date column is datetime
    if df_pl[date_col].dtype != pl.Datetime:
        df_pl = df_pl.with_columns(pl.col(date_col).cast(pl.Date).cast(pl.Datetime))

    # 1. Base date parts
    df_pl = df_pl.with_columns(
        [
            pl.col(date_col).dt.weekday().alias("dayofweek"),
            pl.col(date_col).dt.day().alias("day"),
            pl.col(date_col).dt.month().alias("month"),
            pl.col(date_col).dt.year().alias("year"),
            pl.col(date_col).dt.week().alias("week"),
            (pl.col(date_col).dt.weekday().is_in([6, 7])).cast(pl.Int8).alias("is_weekend"),
        ]
    )

    # 2. Cyclical sine and cosine transformations
    df_pl = df_pl.with_columns(
        [
            (2 * np.pi * pl.col("dayofweek") / 7).sin().alias("dow_sin"),
            (2 * np.pi * pl.col("dayofweek") / 7).cos().alias("dow_cos"),
            (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
            (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos"),
            (2 * np.pi * pl.col("day") / 31).sin().alias("day_sin"),
            (2 * np.pi * pl.col("day") / 31).cos().alias("day_cos"),
        ]
    )

    # 3. State-aware SNAP active flag
    snap_cols = [c for c in df_pl.columns if c.startswith("snap_")]
    if snap_cols and state_col in df_pl.columns:
        df_pl = df_pl.with_columns(
            pl.when((pl.col(state_col) == "CA") & (pl.col("snap_CA") == 1))
            .then(1)
            .when((pl.col(state_col) == "TX") & (pl.col("snap_TX") == 1))
            .then(1)
            .when((pl.col(state_col) == "WI") & (pl.col("snap_WI") == 1))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("active_snap")
        )

    # 4. Event indicators
    if "event_name_1" in df_pl.columns:
        df_pl = df_pl.with_columns(
            [
                pl.col("event_name_1").is_not_null().cast(pl.Int8).alias("is_event_1"),
                pl.col("event_name_1").fill_null("None").cast(pl.Utf8),
                pl.col("event_type_1").fill_null("None").cast(pl.Utf8)
                if "event_type_1" in df_pl.columns
                else pl.lit("None"),
            ]
        )
    if "event_name_2" in df_pl.columns:
        df_pl = df_pl.with_columns(
            [
                pl.col("event_name_2").is_not_null().cast(pl.Int8).alias("is_event_2"),
                pl.col("event_name_2").fill_null("None").cast(pl.Utf8),
                pl.col("event_type_2").fill_null("None").cast(pl.Utf8)
                if "event_type_2" in df_pl.columns
                else pl.lit("None"),
            ]
        )

    return reduce_mem_usage(df_pl.to_pandas())
