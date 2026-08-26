"""Leakage-free lag, rolling window, and exponential moving average feature engineering."""

from typing import List, Optional, Union

import pandas as pd
import polars as pl

from src.data.loader import reduce_mem_usage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_lag_and_rolling_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    base_shift: int = 28,
    target_col: str = "sales",
    group_col: str = "id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Extract strictly leakage-free lag and rolling features.

    To ensure zero target leakage across a 28-day forecasting horizon, all rolling
    statistics are computed on top of the series shifted by at least `base_shift` (28 days).
    """
    if lags is None:
        lags = [28, 35, 42, 49, 56, 63, 70]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28, 60, 90, 180]

    logger.info(
        f"Extracting lag features ({lags}) and rolling windows ({rolling_windows}) "
        f"with base_shift={base_shift} on {group_col}..."
    )

    df_pl = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
    df_pl = df_pl.sort([group_col, date_col])

    # If target_col not present (e.g. inference payload), create zero column
    if target_col not in df_pl.columns:
        df_pl = df_pl.with_columns(pl.lit(0.0).cast(pl.Float32).alias(target_col))

    # 1. Discrete shifted lag expressions & rolling stats
    lag_exprs: List[pl.Expr] = []
    for lag in lags:
        lag_exprs.append(
            pl.col(target_col).shift(lag).over(group_col).cast(pl.Float32).alias(f"sales_lag_{lag}")
        )

    shifted_target = pl.col(target_col).shift(base_shift)
    for window in rolling_windows:
        lag_exprs.append(
            shifted_target.rolling_mean(window_size=window)
            .over(group_col)
            .cast(pl.Float32)
            .alias(f"rolling_mean_{window}_lag_{base_shift}")
        )
        lag_exprs.append(
            shifted_target.rolling_std(window_size=window)
            .over(group_col)
            .cast(pl.Float32)
            .alias(f"rolling_std_{window}_lag_{base_shift}")
        )
        if window in [7, 28]:
            lag_exprs.append(
                shifted_target.rolling_max(window_size=window)
                .over(group_col)
                .cast(pl.Float32)
                .alias(f"rolling_max_{window}_lag_{base_shift}")
            )
            lag_exprs.append(
                shifted_target.rolling_min(window_size=window)
                .over(group_col)
                .cast(pl.Float32)
                .alias(f"rolling_min_{window}_lag_{base_shift}")
            )

    df_featured = df_pl.with_columns(lag_exprs)

    # 2. Ratio / Relative momentum features
    ratio_exprs = []
    lag_col_name = f"sales_lag_{base_shift}"
    rm7_col_name = f"rolling_mean_7_lag_{base_shift}"
    rm28_col_name = f"rolling_mean_28_lag_{base_shift}"

    if lag_col_name in df_featured.columns and rm7_col_name in df_featured.columns:
        ratio_exprs.append(
            (pl.col(lag_col_name) / (pl.col(rm7_col_name) + 1e-4))
            .cast(pl.Float32)
            .alias("lag28_to_rolling_mean_7_ratio")
        )
    if rm7_col_name in df_featured.columns and rm28_col_name in df_featured.columns:
        ratio_exprs.append(
            (pl.col(rm7_col_name) / (pl.col(rm28_col_name) + 1e-4))
            .cast(pl.Float32)
            .alias("rolling_mean_7_to_28_ratio")
        )

    if ratio_exprs:
        df_featured = df_featured.with_columns(ratio_exprs)

    return reduce_mem_usage(df_featured.to_pandas())
