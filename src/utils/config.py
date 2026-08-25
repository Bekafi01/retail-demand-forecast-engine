"""Configuration loader and schema definition."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    external_data_dir: str = "data/external"
    models_dir: str = "models"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    raw_calendar_file: str = "data/raw/calendar.csv"
    raw_prices_file: str = "data/raw/sell_prices.csv"
    raw_sales_train_file: str = "data/raw/sales_train_evaluation.csv"
    processed_sales_file: str = "data/processed/sales_long.parquet"
    processed_features_file: str = "data/processed/features_df.parquet"


class ForecastConfig(BaseModel):
    horizon: int = 28
    frequency: str = "D"
    target_col: str = "sales"
    id_cols: List[str] = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]
    series_id_col: str = "id"
    date_col: str = "date"
    day_idx_col: str = "d"


class HierarchyConfig(BaseModel):
    levels: Dict[int, Union[str, List[str]]] = Field(default_factory=dict)


class DataSplitConfig(BaseModel):
    train_start_d: int = 1
    train_end_d: int = 1913
    val_start_d: int = 1914
    val_end_d: int = 1941
    test_start_d: int = 1942
    test_end_d: int = 1969
    backtest_windows: int = 3
    backtest_step_size: int = 28


class FeaturesConfig(BaseModel):
    lags: List[int] = [28, 35, 42, 49, 56, 63, 70]
    recursive_lags: List[int] = [1, 2, 3, 7, 14, 21, 28]
    rolling_windows: List[int] = [7, 14, 28, 60, 90, 180]
    rolling_stats: List[str] = ["mean", "std", "min", "max", "skew"]
    ewma_alphas: List[float] = [0.1, 0.3, 0.5]
    calendar_features: List[str] = Field(default_factory=list)
    price_features: List[str] = Field(default_factory=list)
    categorical_cols: List[str] = Field(default_factory=list)


class MLOpsConfig(BaseModel):
    experiment_name: str = "retail_demand_forecast"
    tracking_uri: str = "mlruns"
    model_registry_name: str = "m5_retail_demand_champion"
    metric_for_promotion: str = "wrmsse"
    promotion_threshold_improvement: float = 0.01


class MonitoringConfig(BaseModel):
    psi: Dict[str, float] = Field(
        default_factory=lambda: {"warning_threshold": 0.1, "critical_threshold": 0.2}
    )
    ks_test: Dict[str, float] = Field(default_factory=lambda: {"alpha": 0.05})
    error_tracking: Dict[str, float] = Field(
        default_factory=lambda: {"wape_alert_threshold": 0.25, "rmsse_alert_threshold": 1.20}
    )


class AppConfig(BaseModel):
    project: Dict[str, Any] = Field(
        default_factory=lambda: {
            "name": "retail-demand-forecast-engine",
            "version": "0.1.0",
            "random_seed": 42,
        }
    )
    paths: PathsConfig = Field(default_factory=PathsConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    data_split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: Dict[str, Any] = Field(default_factory=dict)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load configuration from YAML file or return default configuration."""
    if config_path is None:
        # Default config lookup
        candidates = [
            Path("configs/model_config.yaml"),
            Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return AppConfig(**raw_data)

    return AppConfig()
