"""End-to-end training, backtesting, and model registry orchestration pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from mlops.registry.manager import ModelRegistryManager
from mlops.tracking.tracker import ExperimentTracker
from src.data.loader import generate_synthetic_m5_data, load_raw_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.backtest import run_backtest
from src.features.pipeline import build_feature_table
from src.models.gbm import LightGBMForecaster
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train_pipeline")


def run_training_pipeline(
    config_path: str = "configs/model_config.yaml",
    sample_series: int = 150,
) -> None:
    """Execute end-to-end training and promotion pipeline."""
    cfg = load_config(config_path)
    logger.info(f"Loaded configuration for project: {cfg.project.get('name')}")

    # 1. Ingestion
    raw_dir = Path(cfg.paths.raw_data_dir)
    try:
        cal_df, prc_df, sal_df = load_raw_data(raw_dir, nrows=sample_series)
        logger.info(f"Loaded {len(sal_df)} series from {raw_dir}")
    except Exception as e:
        logger.warning(f"Raw data not found ({e}). Generating synthetic M5 dataset...")
        cal_df, prc_df, sal_df = generate_synthetic_m5_data(
            num_items=100, num_stores=4, num_days=365, random_seed=42
        )

    # 2. Preprocessing & Feature Store
    sales_long = melt_sales_data(sal_df)
    merged_df = merge_calendar_and_prices(sales_long, cal_df, prc_df)
    feat_df = build_feature_table(merged_df)
    logger.info(
        f"Feature table prepared with {len(feat_df):,} rows and {len(feat_df.columns)} columns."
    )

    # 3. Model & MLflow Tracking
    tracker = ExperimentTracker(experiment_name=cfg.project.get("name", "retail-forecast"))
    registry = ModelRegistryManager(model_name="retail-demand-champion")

    with tracker.start_run(run_name="LightGBM_Tweedie_Pipeline"):
        model = LightGBMForecaster(
            n_estimators=cfg.models.lightgbm.n_estimators,
            learning_rate=cfg.models.lightgbm.learning_rate,
            num_leaves=cfg.models.lightgbm.num_leaves,
            tweedie_variance_power=cfg.models.lightgbm.tweedie_variance_power,
        )

        # 4. Temporal Backtesting
        agg_metrics, oof_df, fold_summaries = run_backtest(
            model=model,
            df=feat_df,
            horizon=cfg.forecast.horizon,
            n_splits=cfg.forecast.backtest_splits,
        )

        # Log parameters and metrics to MLflow
        tracker.log_params(model.get_params())
        tracker.log_metrics(agg_metrics)
        tracker.log_model(model, artifact_path="champion_model")

        # 5. Model Registry Evaluation & Promotion
        run_id = tracker.active_run.info.run_id
        version = registry.register_model_version(run_id=run_id, artifact_path="champion_model")
        promoted, promo_msg = registry.evaluate_and_promote(
            new_version=version,
            new_wrmsse=agg_metrics["mean_wrmsse"],
        )
        logger.info(f"Promotion Result: {promo_msg}")

    logger.info("Training pipeline completed successfully! 🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Demand Forecast Training Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/model_config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--sample-series", type=int, default=150, help="Number of series to train on"
    )
    args = parser.parse_args()

    run_training_pipeline(config_path=args.config, sample_series=args.sample_series)
