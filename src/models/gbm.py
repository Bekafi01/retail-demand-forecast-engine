"""Gradient Boosted Decision Tree (GBDT) forecasters: LightGBM and CatBoost with Tweedie objective."""

from typing import Any, Dict, List, Optional, Union
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.features.pipeline import build_feature_table, get_feature_column_names
from src.models.base import BaseDemandForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMForecaster(BaseDemandForecaster):
    """LightGBM forecaster with Tweedie loss, categorical features, and SHAP explainability."""

    def __init__(
        self,
        objective: str = "tweedie",
        tweedie_variance_power: float = 1.15,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        n_estimators: int = 300,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 1,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        **params: Any,
    ):
        super().__init__(
            name="LightGBM_Tweedie",
            objective=objective,
            tweedie_variance_power=tweedie_variance_power,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            random_state=random_state,
            **params,
        )
        self.objective = objective
        self.tweedie_variance_power = tweedie_variance_power
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.random_state = random_state
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.model: Optional[lgb.Booster] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return model parameters for MLflow logging and hyperparameter tuning."""
        return {
            "name": self.name,
            "objective": self.objective,
            "tweedie_variance_power": self.tweedie_variance_power,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "n_estimators": self.n_estimators,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "random_state": self.random_state,
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        val_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "LightGBMForecaster":
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_col = series_id_col

        # 1. Check if features are already computed, otherwise compute feature table
        if "sales_lag_28" not in train_df.columns:
            logger.info("Computing feature table for training data...")
            full_train = build_feature_table(train_df, target_col=target_col, date_col=date_col)
        else:
            full_train = train_df.copy()

        # Drop initial rows with NaN from shifted lags
        full_train = full_train.dropna(subset=["sales_lag_28"]).reset_index(drop=True)

        if self.feature_cols is None:
            self.feature_cols = get_feature_column_names(full_train, target_col=target_col)

        if self.categorical_cols is None:
            self.categorical_cols = [
                col
                for col in self.feature_cols
                if full_train[col].dtype == "category" or str(full_train[col].dtype) == "object"
            ]

        # Ensure categoricals are properly typed
        for col in self.categorical_cols:
            if col in full_train.columns:
                full_train[col] = full_train[col].astype("category")

        X_train = full_train[self.feature_cols]
        y_train = full_train[target_col].values

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.categorical_cols,
            free_raw_data=False,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if val_df is not None:
            if "sales_lag_28" not in val_df.columns:
                full_val = build_feature_table(val_df, target_col=target_col, date_col=date_col)
            else:
                full_val = val_df.copy()
            full_val = full_val.dropna(subset=["sales_lag_28"]).reset_index(drop=True)
            if not full_val.empty:
                for col in self.categorical_cols:
                    if col in full_val.columns:
                        full_val[col] = full_val[col].astype("category")

                val_data = lgb.Dataset(
                    full_val[self.feature_cols],
                    label=full_val[target_col].values,
                    categorical_feature=self.categorical_cols,
                    reference=train_data,
                    free_raw_data=False,
                )
                valid_sets.append(val_data)
                valid_names.append("val")

        lgb_params = {
            "objective": self.objective,
            "tweedie_variance_power": self.tweedie_variance_power,
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "random_state": self.random_state,
            "verbose": -1,
        }

        callbacks = []
        if len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

        self.model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks if callbacks else None,
        )

        self.is_fitted = True
        return self

    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.model is None or self.feature_cols is None:
            raise ValueError("Model must be fitted before predict() is called.")

        if "sales_lag_28" not in pred_df.columns:
            features_df = build_feature_table(pred_df, target_col=self.target_col, date_col=self.date_col)
        else:
            features_df = pred_df.copy()

        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0

        # Ensure all categorical columns match category dtype
        if self.categorical_cols:
            for col in self.categorical_cols:
                if col in features_df.columns:
                    features_df[col] = features_df[col].astype("category")

        X_pred = features_df[self.feature_cols]
        raw_preds = self.model.predict(X_pred)

        res = pred_df[[self.series_id_col, self.date_col]].copy()
        res["y_pred"] = np.maximum(0.0, raw_preds.astype(np.float32))
        return res

    def get_feature_importances(self, importance_type: str = "gain") -> pd.DataFrame:
        """Extract sorted feature importances from fitted LightGBM model."""
        if not self.is_fitted or self.model is None or self.feature_cols is None:
            raise ValueError("Model must be fitted before getting feature importances.")

        imp = self.model.feature_importance(importance_type=importance_type)
        df_imp = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp


class CatBoostForecaster(BaseDemandForecaster):
    """CatBoost forecaster with native categorical support and Tweedie loss."""

    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.06,
        depth: int = 6,
        loss_function: str = "Tweedie:variance_power=1.15",
        random_seed: int = 42,
        **params: Any,
    ):
        super().__init__(
            name="CatBoost_Tweedie",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function=loss_function,
            random_seed=random_seed,
            **params,
        )
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.feature_cols: Optional[List[str]] = None
        self.categorical_cols: Optional[List[str]] = None
        self.model: Optional[CatBoostRegressor] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return model parameters for MLflow logging and hyperparameter tuning."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "loss_function": self.loss_function,
            "random_seed": self.random_seed,
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_col: str = "id",
        **kwargs: Any,
    ) -> "CatBoostForecaster":
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_col = series_id_col

        if "sales_lag_28" not in train_df.columns:
            full_train = build_feature_table(train_df, target_col=target_col, date_col=date_col)
        else:
            full_train = train_df.copy()

        full_train = full_train.dropna(subset=["sales_lag_28"]).reset_index(drop=True)
        self.feature_cols = get_feature_column_names(full_train, target_col=target_col)
        self.categorical_cols = [
            col
            for col in self.feature_cols
            if full_train[col].dtype == "category" or str(full_train[col].dtype) == "object"
        ]

        X_train = full_train[self.feature_cols].copy()
        for c in self.categorical_cols:
            X_train[c] = X_train[c].astype(str)

        y_train = full_train[target_col].values

        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=False,
        )
        self.model.fit(X_train, y_train, cat_features=self.categorical_cols)
        self.is_fitted = True
        return self

    def predict(
        self,
        pred_df: pd.DataFrame,
        horizon: int = 28,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.model is None or self.feature_cols is None:
            raise ValueError("Model must be fitted before predict() is called.")

        if "sales_lag_28" not in pred_df.columns:
            features_df = build_feature_table(pred_df, target_col=self.target_col, date_col=self.date_col)
        else:
            features_df = pred_df.copy()

        X_pred = features_df[self.feature_cols].copy()
        if self.categorical_cols:
            for c in self.categorical_cols:
                X_pred[c] = X_pred[c].astype(str)

        raw_preds = self.model.predict(X_pred)
        res = pred_df[[self.series_id_col, self.date_col]].copy()
        res["y_pred"] = np.maximum(0.0, raw_preds.astype(np.float32))
        return res
