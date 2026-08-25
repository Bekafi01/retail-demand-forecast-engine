"""Statistical and intermittent demand forecasting models using StatsForecast."""

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
)

from src.models.base import BaseDemandForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StatsForecastBase(BaseDemandForecaster):
    """Base adapter connecting StatsForecast models to the BaseDemandForecaster interface."""

    def __init__(
        self,
        name: str,
        models: List[Any],
        n_jobs: int = -1,
        max_history_days: int = 365,
        **params: Any,
    ):
        super().__init__(name=name, n_jobs=n_jobs, max_history_days=max_history_days, **params)
        self.models = models
        self.n_jobs = n_jobs
        self.max_history_days = max_history_days
        self.sf: Optional[StatsForecast] = None
        self.target_col = "sales"
        self.date_col = "date"
        self.series_id_col = "id"

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        **kwargs: Any,
    ) -> "StatsForecastBase":
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_col = series_id_col

        # Transform to StatsForecast standard schema: unique_id, ds, y
        sf_df = train_df[[series_id_col, date_col, target_col]].copy()
        sf_df.columns = ["unique_id", "ds", "y"]
        sf_df["unique_id"] = sf_df["unique_id"].astype(str)
        sf_df["ds"] = pd.to_datetime(sf_df["ds"])
        sf_df["y"] = sf_df["y"].astype(np.float32)

        # Sort chronologically per series and slice most recent history to ensure fast fitting
        sf_df = sf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        if self.max_history_days and self.max_history_days > 0:
            sf_df = (
                sf_df.groupby("unique_id", observed=False)
                .tail(self.max_history_days)
                .reset_index(drop=True)
            )

        self.sf = StatsForecast(
            models=self.models,
            freq="D",
            n_jobs=self.n_jobs,
        )
        self.sf.fit(sf_df)
        self.is_fitted = True
        return self

    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.sf is None:
            raise ValueError("Model must be fitted before calling predict().")

        # Generate forecasts from StatsForecast
        sf_preds = self.sf.predict(h=horizon).reset_index()

        # The forecast column is the name of the first model in self.models
        model_col = self.models[0].__class__.__name__
        if model_col not in sf_preds.columns:
            num_cols = [c for c in sf_preds.columns if c not in ["unique_id", "ds"]]
            model_col = num_cols[0]

        # Map back to [series_id_col, date_col, 'y_pred']
        pred_sf_df = sf_preds[["unique_id", "ds", model_col]].rename(
            columns={"unique_id": self.series_id_col, "ds": self.date_col, model_col: "y_pred"}
        )
        pred_sf_df[self.series_id_col] = pred_sf_df[self.series_id_col].astype(str)
        pred_sf_df[self.date_col] = pd.to_datetime(pred_sf_df[self.date_col])

        out = pred_df[[self.series_id_col, self.date_col]].copy()
        out[self.series_id_col] = out[self.series_id_col].astype(str)
        out[self.date_col] = pd.to_datetime(out[self.date_col])

        merged = out.merge(pred_sf_df, on=[self.series_id_col, self.date_col], how="left")
        merged["y_pred"] = merged["y_pred"].fillna(0.0)
        merged["y_pred"] = np.maximum(0.0, merged["y_pred"].astype(np.float32))
        return merged

    def predict_intervals(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        quantiles: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        if not self.is_fitted or self.sf is None:
            raise ValueError("Model must be fitted before calling predict_intervals().")

        levels = []
        for q in quantiles:
            if q != 0.5:
                lvl = int(round(abs(q - 0.5) * 200))
                if 0 < lvl < 100 and lvl not in levels:
                    levels.append(lvl)

        try:
            sf_preds = self.sf.predict(h=horizon, level=levels if levels else None).reset_index()
            model_col = self.models[0].__class__.__name__

            out = pred_df[[self.series_id_col, self.date_col]].copy()
            out[self.series_id_col] = out[self.series_id_col].astype(str)
            out[self.date_col] = pd.to_datetime(out[self.date_col])

            sf_preds["unique_id"] = sf_preds["unique_id"].astype(str)
            sf_preds["ds"] = pd.to_datetime(sf_preds["ds"])

            res = out.merge(
                sf_preds,
                left_on=[self.series_id_col, self.date_col],
                right_on=["unique_id", "ds"],
                how="left",
            )
            res["y_pred"] = np.maximum(0.0, res[model_col].fillna(0.0).astype(np.float32))

            for q in quantiles:
                col_name = f"q_{int(q * 100)}"
                if q == 0.5:
                    res[col_name] = res["y_pred"]
                else:
                    lvl = int(round(abs(q - 0.5) * 200))
                    bound_type = "lo" if q < 0.5 else "hi"
                    level_col = f"{model_col}-{bound_type}-{lvl}"
                    if level_col in res.columns:
                        res[col_name] = np.maximum(0.0, res[level_col].fillna(res["y_pred"]))
                    else:
                        mult = 0.8 + 0.4 * q if q < 0.5 else 1.0 + 0.4 * (q - 0.5)
                        res[col_name] = np.maximum(0.0, res["y_pred"] * mult)

            keep_cols = [self.series_id_col, self.date_col, "y_pred"] + [
                f"q_{int(q * 100)}" for q in quantiles
            ]
            return res[keep_cols]
        except Exception as e:
            logger.warning(
                f"Statistical interval prediction failed with error ({e}), falling back to base interval generator."
            )
            return super().predict_intervals(
                pred_df, horizon=horizon, quantiles=quantiles, **kwargs
            )


class CrostonForecaster(StatsForecastBase):
    """Croston Method Forecaster for intermittent and zero-inflated retail demand.

    Variants:
        - "sba": Syntetos-Boylan Approximation (defacto standard, corrects positive bias of classic Croston)
        - "classic": Classic Croston method
        - "optimized": Croston with parameter optimization
    """

    def __init__(
        self,
        variant: str = "sba",
        n_jobs: int = -1,
        max_history_days: int = 365,
        **params: Any,
    ):
        self.variant = variant.lower()
        if self.variant == "classic":
            model = CrostonClassic()
        elif self.variant == "optimized":
            model = CrostonOptimized()
        else:
            model = CrostonSBA()

        super().__init__(
            name=f"Croston_{self.variant.upper()}",
            models=[model],
            n_jobs=n_jobs,
            max_history_days=max_history_days,
            **params,
        )


class AutoThetaForecaster(StatsForecastBase):
    """AutoTheta Forecaster with seasonal decomposition (7-day standard retail period)."""

    def __init__(
        self,
        season_length: int = 7,
        n_jobs: int = -1,
        max_history_days: int = 365,
        **params: Any,
    ):
        super().__init__(
            name=f"AutoTheta_{season_length}d",
            models=[AutoTheta(season_length=season_length)],
            n_jobs=n_jobs,
            season_length=season_length,
            max_history_days=max_history_days,
            **params,
        )


class AutoETSForecaster(StatsForecastBase):
    """Automated Exponential Smoothing State-Space (ETS) Forecaster."""

    def __init__(
        self,
        season_length: int = 7,
        model: str = "ZZZ",
        n_jobs: int = -1,
        max_history_days: int = 365,
        **params: Any,
    ):
        super().__init__(
            name=f"AutoETS_{season_length}d",
            models=[AutoETS(season_length=season_length, model=model)],
            n_jobs=n_jobs,
            season_length=season_length,
            model=model,
            max_history_days=max_history_days,
            **params,
        )


class AutoARIMAForecaster(StatsForecastBase):
    """Automated Seasonal ARIMA Forecaster with optimal order selection."""

    def __init__(
        self,
        season_length: int = 7,
        max_p: int = 2,
        max_q: int = 2,
        max_d: int = 1,
        n_jobs: int = -1,
        max_history_days: int = 180,
        **params: Any,
    ):
        super().__init__(
            name=f"AutoARIMA_{season_length}d",
            models=[
                AutoARIMA(
                    season_length=season_length,
                    max_p=max_p,
                    max_q=max_q,
                    max_d=max_d,
                    stepwise=True,
                )
            ],
            n_jobs=n_jobs,
            season_length=season_length,
            max_history_days=max_history_days,
            **params,
        )
