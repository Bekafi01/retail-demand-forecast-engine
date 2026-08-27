"""Deep Learning and Multi-Layer Perceptron (MLP) demand forecasting architectures."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.features.pipeline import build_feature_table, get_feature_column_names
from src.models.base import BaseDemandForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPDemandForecaster(BaseDemandForecaster):
    """Deep Multi-Layer Perceptron (MLP) neural network forecaster for multi-horizon demand estimation."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        max_iter: int = 150,
        batch_size: Union[int, str] = 128,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        **params: Any,
    ):
        super().__init__(
            name="MLP_NeuralForecaster",
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            batch_size=batch_size,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            **params,
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.feature_cols = feature_cols

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[MLPRegressor] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return neural network hyperparameters."""
        return {
            "name": self.name,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "random_state": self.random_state,
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        **kwargs: Any,
    ) -> "MLPDemandForecaster":
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_col = series_id_col

        if "sales_lag_28" not in train_df.columns:
            full_train = build_feature_table(train_df, target_col=target_col, date_col=date_col)
        else:
            full_train = train_df.copy()

        full_train = full_train.dropna(subset=["sales_lag_28"]).reset_index(drop=True)

        if self.feature_cols is None:
            all_cols = get_feature_column_names(full_train, target_col=target_col)
            self.feature_cols = [
                col for col in all_cols if pd.api.types.is_numeric_dtype(full_train[col])
            ]

        if len(full_train) == 0:
            raise ValueError("Training DataFrame contains 0 samples after removing NaN lags.")

        X_train = full_train[self.feature_cols].fillna(0.0).values
        y_train = full_train[target_col].values

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        logger.info(
            f"Training MLP neural network {self.hidden_layer_sizes} on {len(X_train_scaled):,} samples x {len(self.feature_cols)} features..."
        )

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            validation_fraction=0.1,
            solver="adam",
        )

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if (
            not self.is_fitted
            or self.model is None
            or self.scaler is None
            or self.feature_cols is None
        ):
            raise ValueError("Model must be fitted before predict() is called.")

        res = pred_df[[self.series_id_col, self.date_col]].copy()
        if len(pred_df) == 0:
            res["y_pred"] = np.array([], dtype=np.float32)
            return res

        if "sales_lag_28" not in pred_df.columns:
            features_df = build_feature_table(
                pred_df, target_col=self.target_col, date_col=self.date_col
            )
        else:
            features_df = pred_df.copy()

        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0

        X_pred = features_df[self.feature_cols].fillna(0.0).values
        X_pred_scaled = self.scaler.transform(X_pred)

        raw_preds = self.model.predict(X_pred_scaled)
        res["y_pred"] = np.maximum(0.0, raw_preds.astype(np.float32))
        return res

    @property
    def loss_curve(self) -> List[float]:
        """Return training loss progression."""
        if not self.is_fitted or self.model is None or not hasattr(self.model, "loss_curve_"):
            return []
        return list(self.model.loss_curve_)
