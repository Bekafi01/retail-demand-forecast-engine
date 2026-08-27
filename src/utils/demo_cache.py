"""Fast pre-caching and retrieval of demo data and models for instantaneous Streamlit UI loading."""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.conformal import ConformalCalibrator
from src.features.pipeline import build_feature_table
from src.models.gbm import LightGBMForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_or_create_demo_cache(
    cache_dir: Path,
    num_items: int = 30,
    num_stores: int = 3,
    num_days: int = 180,
) -> Tuple[pd.DataFrame, pd.DataFrame, LightGBMForecaster]:
    """Retrieve pre-built demo feature store, validated forecasts, and trained model for sub-second UI startup.

    If cached files are present in cache_dir, loads them in milliseconds.
    Otherwise, generates and persists them once.
    """
    cache_dir = Path(cache_dir)
    processed_dir = cache_dir / "data" / "processed"
    models_dir = cache_dir / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    featured_path = processed_dir / "demo_featured.parquet"
    forecast_path = processed_dir / "demo_forecast.parquet"
    model_path = models_dir / "demo_champion.joblib"

    if featured_path.exists() and forecast_path.exists() and model_path.exists():
        try:
            featured_df = pd.read_parquet(featured_path)
            forecast_df = pd.read_parquet(forecast_path)
            model = joblib.load(model_path)
            return featured_df, forecast_df, model
        except Exception as e:
            logger.warning(f"Error loading demo cache ({e}). Regenerating cache...")

    logger.info("Generating optimized demo dataset for instantaneous dashboard loading...")
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=num_items, num_stores=num_stores, num_days=num_days, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    merged = merge_calendar_and_prices(sales_long, cal, prc)
    featured_df = build_feature_table(merged)

    dates = sorted(featured_df["date"].unique())
    train_df = featured_df[featured_df["date"] < dates[-28]]
    val_df = featured_df[featured_df["date"] >= dates[-28]]

    model = LightGBMForecaster(n_estimators=60, learning_rate=0.08)
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    conformal = ConformalCalibrator(normalized=True)
    conformal.fit(train_df["sales"].values[-len(preds) :], preds["y_pred"].values)
    intervals_df = conformal.predict_intervals(preds, alphas=[0.1, 0.2])

    forecast_df = val_df.merge(
        intervals_df[["id", "date", "y_pred", "lower_90", "upper_90"]], on=["id", "date"]
    )

    # Persist for instant sub-second reuse
    featured_df.to_parquet(featured_path, index=False)
    forecast_df.to_parquet(forecast_path, index=False)
    joblib.dump(model, model_path)
    logger.info("Demo feature store and champion model persisted successfully!")

    return featured_df, forecast_df, model
