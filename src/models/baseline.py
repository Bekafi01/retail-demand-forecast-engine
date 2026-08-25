"""Baseline forecasting models: Naive, Seasonal Naive, Moving Average, Exponential Smoothing."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.models.base import BaseDemandForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NaiveForecaster(BaseDemandForecaster):
    """Last-known-value Naive Forecaster: y_pred(t+h) = y(t)."""

    def __init__(self, **params: Any):
        super().__init__(name="Naive", **params)
        self.last_values: Dict[str, float] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        **kwargs: Any,
    ) -> "NaiveForecaster":
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_col = series_id_col

        sorted_df = train_df.sort_values([series_id_col, date_col])
        last_records = sorted_df.groupby(series_id_col, observed=False).last()
        self.last_values = {str(k): float(v) for k, v in last_records[target_col].items()}
        self.is_fitted = True
        return self

    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict() is called.")

        res = pred_df[[self.series_id_col, self.date_col]].copy()
        sids = res[self.series_id_col].astype(str)
        mapped = sids.map(self.last_values).fillna(0.0).values
        res["y_pred"] = np.maximum(0.0, mapped.astype(np.float32))
        return res


class SeasonalNaiveForecaster(BaseDemandForecaster):
    """Seasonal Naive Forecaster: y_pred(t+h) = y(t + h - seasonal_period)."""

    def __init__(self, season_length: int = 7, **params: Any):
        super().__init__(
            name=f"SeasonalNaive_{season_length}d", season_length=season_length, **params
        )
        self.season_length = season_length
        self.seasonal_patterns: Dict[str, np.ndarray] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        **kwargs: Any,
    ) -> "SeasonalNaiveForecaster":
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_col = series_id_col

        sorted_df = train_df.sort_values([series_id_col, date_col])
        for sid, grp in sorted_df.groupby(series_id_col, observed=False):
            y = grp[target_col].values
            pattern = (
                y[-self.season_length :]
                if len(y) >= self.season_length
                else np.pad(y, (self.season_length - len(y), 0))
            )
            self.seasonal_patterns[str(sid)] = pattern

        self.is_fitted = True
        return self

    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict() is called.")

        res_list = []
        for sid, grp in pred_df.groupby(self.series_id_col, observed=False):
            pattern = self.seasonal_patterns.get(str(sid), np.zeros(self.season_length))
            n_repeats = (len(grp) // self.season_length) + 1
            tiled = np.tile(pattern, n_repeats)[: len(grp)]
            grp_res = grp[[self.series_id_col, self.date_col]].copy()
            grp_res["y_pred"] = np.maximum(0.0, tiled.astype(np.float32))
            res_list.append(grp_res)

        return pd.concat(res_list, ignore_index=True)


class MovingAverageForecaster(BaseDemandForecaster):
    """Moving Average Forecaster: y_pred(t+h) = mean(y[t-window:t])."""

    def __init__(self, window: int = 28, **params: Any):
        super().__init__(name=f"MovingAverage_{window}d", window=window, **params)
        self.window = window
        self.window_means: Dict[str, float] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        **kwargs: Any,
    ) -> "MovingAverageForecaster":
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_col = series_id_col

        sorted_df = train_df.sort_values([series_id_col, date_col])
        for sid, grp in sorted_df.groupby(series_id_col, observed=False):
            y = grp[target_col].values
            window_slice = y[-self.window :] if len(y) >= self.window else y
            mean_val = float(np.mean(window_slice)) if len(window_slice) > 0 else 0.0
            self.window_means[str(sid)] = mean_val

        self.is_fitted = True
        return self

    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict() is called.")

        res = pred_df[[self.series_id_col, self.date_col]].copy()
        sids = res[self.series_id_col].astype(str)
        mapped = sids.map(self.window_means).fillna(0.0).values
        res["y_pred"] = np.maximum(0.0, mapped.astype(np.float32))
        return res


class ExponentialSmoothingForecaster(BaseDemandForecaster):
    """Simple Exponential Smoothing Forecaster with level estimation."""

    def __init__(self, alpha: float = 0.2, **params: Any):
        super().__init__(name=f"ExpSmoothing_a{alpha}", alpha=alpha, **params)
        self.alpha = alpha
        self.levels: Dict[str, float] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        **kwargs: Any,
    ) -> "ExponentialSmoothingForecaster":
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_col = series_id_col

        sorted_df = train_df.sort_values([series_id_col, date_col])
        for sid, grp in sorted_df.groupby(series_id_col, observed=False):
            y = grp[target_col].values
            if len(y) == 0:
                self.levels[str(sid)] = 0.0
                continue

            level = float(y[0])
            for val in y[1:]:
                level = self.alpha * float(val) + (1.0 - self.alpha) * level
            self.levels[str(sid)] = level

        self.is_fitted = True
        return self

    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict() is called.")

        res = pred_df[[self.series_id_col, self.date_col]].copy()
        sids = res[self.series_id_col].astype(str)
        mapped = sids.map(self.levels).fillna(0.0).values
        res["y_pred"] = np.maximum(0.0, mapped.astype(np.float32))
        return res
