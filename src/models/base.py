"""Abstract base class for all demand forecasting models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd


class BaseDemandForecaster(ABC):
    """Unified interface for point and interval demand forecasting."""

    def __init__(self, name: str = "base_forecaster", **params: Any):
        self.name = name
        self.params = params
        self.is_fitted = False

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters (scikit-learn estimator compatible)."""
        params = {"name": self.name}
        params.update(self.params)
        return params

    def set_params(self, **params: Any) -> "BaseDemandForecaster":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            self.params[key] = value
        return self

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        **kwargs: Any,
    ) -> "BaseDemandForecaster":
        """Fit model to training dataset."""
        pass

    @abstractmethod
    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate point forecasts for the future horizon.

        Returns:
            DataFrame containing ['id', date_col, 'y_pred']
        """
        pass

    def predict_intervals(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        quantiles: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate prediction intervals (e.g. P10, P50, P90).

        Default fallback uses point forecast as P50 and empirical normal error margin.
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        point_preds = self.predict(pred_df, horizon=horizon, **kwargs)
        res = point_preds.copy()

        # Simple empirical dispersion fallback for baselines
        for q in quantiles:
            col_name = f"q_{int(q * 100)}"
            if q == 0.5:
                res[col_name] = res["y_pred"]
            elif q < 0.5:
                res[col_name] = np.maximum(0.0, res["y_pred"] * (0.8 + 0.4 * q))
            else:
                res[col_name] = res["y_pred"] * (1.0 + 0.4 * (q - 0.5))

        return res

    def save(self, path: Union[str, Path]) -> None:
        """Persist model instance to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseDemandForecaster":
        """Load persisted model from disk."""
        return joblib.load(path)
