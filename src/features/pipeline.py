"""End-to-end feature pipeline orchestrating temporal, lag, rolling, and price features."""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from src.data.loader import reduce_mem_usage
from src.features.calendar import extract_calendar_features
from src.features.lags import extract_lag_and_rolling_features
from src.features.price import extract_price_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_feature_table(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    base_shift: int = 28,
    target_col: str = "sales",
    group_col: str = "id",
    date_col: str = "date",
    price_col: str = "sell_price",
    categorical_cols: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Execute complete feature engineering pipeline on merged sales DataFrame."""
    logger.info("Executing end-to-end feature engineering pipeline...")

    if categorical_cols is None:
        categorical_cols = [
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ]

    # 1. Calendar & Cyclical features
    featured = extract_calendar_features(df, date_col=date_col)

    # 2. Price elasticity & discount features
    featured = extract_price_features(featured, price_col=price_col, date_col=date_col)

    # 3. Leak-free Lags & Rolling statistics
    featured = extract_lag_and_rolling_features(
        featured,
        lags=lags,
        rolling_windows=rolling_windows,
        base_shift=base_shift,
        target_col=target_col,
        group_col=group_col,
        date_col=date_col,
    )

    # 4. Convert categoricals to category dtype for LightGBM/CatBoost
    for col in categorical_cols:
        if col in featured.columns:
            featured[col] = featured[col].astype("category")

    # 5. Clean up memory
    featured = reduce_mem_usage(featured)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Saving feature table ({len(featured):,} rows, {len(featured.columns)} cols) to {out_p}..."
        )
        featured.to_parquet(out_p, index=False, compression="snappy")

    return featured


def get_feature_column_names(
    df: pd.DataFrame,
    target_col: str = "sales",
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """Get list of feature column names excluding metadata, IDs, targets, and dates."""
    if exclude_cols is None:
        exclude_cols = ["id", "date", "d", "wm_yr_wk", target_col]
    return [col for col in df.columns if col not in exclude_cols]
