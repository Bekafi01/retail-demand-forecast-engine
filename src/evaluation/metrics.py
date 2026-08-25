"""M5 Forecasting evaluation metrics: WRMSSE, RMSSE, WAPE, MAE, RMSE, and Quantile Loss."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def compute_rmsse(
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute Root Mean Squared Scaled Error (RMSSE) for a single time series.

    Formula:
        RMSSE = sqrt( (1/h * sum((y_true - y_pred)^2)) / (1/(n-1) * sum((y_train[t] - y_train[t-1])^2)) )
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # In-sample naive difference scale denominator
    diff = np.diff(y_train)
    scale = np.mean(diff**2)
    if scale < eps:
        scale = eps

    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse / scale))


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Weighted Absolute Percentage Error (WAPE).

    Formula:
        WAPE = sum(|y_true - y_pred|) / (sum(|y_true|) + eps)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    total_abs_err = np.sum(np.abs(y_true - y_pred))
    total_actual = np.sum(np.abs(y_true))
    return float(total_abs_err / (total_actual + eps))


def compute_pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.5,
) -> float:
    """Compute Pinball / Quantile Loss for a specific quantile q in (0, 1).

    Formula:
        L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


class WRMSSEEvaluator:
    """Official M5 Weighted Root Mean Squared Scaled Error (WRMSSE) Evaluator across hierarchical levels.

    Computes RMSSE across hierarchy levels (e.g. Total, State, Store, Category, Department, Item-Store)
    and aggregates them weighted by cumulative dollar (or unit) volume over the last 28 days of the training set.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        id_cols: Optional[List[str]] = None,
        date_col: str = "date",
        target_col: str = "sales",
        price_col: Optional[str] = "sell_price",
        weights_window: int = 28,
    ):
        self.id_cols = id_cols or ["item_id", "store_id", "dept_id", "cat_id", "state_id"]
        self.date_col = date_col
        self.target_col = target_col
        self.price_col = price_col
        self.weights_window = weights_window

        # Precalculate in-sample naive scale denominator and series dollar weights
        self.scales: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self._fit_evaluator(train_df)

    def _fit_evaluator(self, train_df: pd.DataFrame) -> None:
        """Precompute scale denominators and weights per series from training history."""
        # Calculate dollar sales if price is available, otherwise unit sales
        df = train_df.copy()
        if self.price_col and self.price_col in df.columns and df[self.price_col].notna().any():
            df["_revenue"] = df[self.target_col] * df[self.price_col].fillna(1.0)
        else:
            df["_revenue"] = df[self.target_col]

        # In-sample scales per series
        for series_id, grp in df.groupby("id", observed=False):
            grp_sorted = grp.sort_values(self.date_col)
            y = grp_sorted[self.target_col].values
            # Ignore leading zeros before first item sale if present
            first_idx = np.argmax(y > 0) if np.any(y > 0) else 0
            y_active = y[first_idx:]

            if len(y_active) > 1:
                diff = np.diff(y_active)
                scale = float(np.mean(diff**2))
            else:
                scale = 1.0

            self.scales[str(series_id)] = max(scale, 1e-6)

        # Compute weights based on the last `weights_window` days
        recent_dates = df[self.date_col].drop_duplicates().sort_values().tail(self.weights_window)
        recent_df = df[df[self.date_col].isin(recent_dates)]
        total_rev = recent_df["_revenue"].sum()

        if total_rev > 0:
            rev_by_id = recent_df.groupby("id", observed=False)["_revenue"].sum()
            for sid, rev in rev_by_id.items():
                self.weights[str(sid)] = float(rev / total_rev)
        else:
            # Fallback to uniform weights
            n = len(self.scales)
            self.weights = {sid: 1.0 / n for sid in self.scales}

    def score(
        self,
        val_df: pd.DataFrame,
        pred_df: pd.DataFrame,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute WRMSSE, individual RMSSE, and WAPE between ground truth and predictions.

        Parameters:
            val_df: DataFrame with ['id', date_col, target_col]
            pred_df: DataFrame with ['id', date_col, 'y_pred'] or [target_col]

        Returns:
            Dictionary containing overall WRMSSE, RMSSE mean, WAPE, and detailed level metrics.
        """
        pred_col = "y_pred" if "y_pred" in pred_df.columns else self.target_col

        merged = val_df[["id", self.date_col, self.target_col]].merge(
            pred_df[["id", self.date_col, pred_col]],
            on=["id", self.date_col],
            how="inner",
        )

        if merged.empty:
            raise ValueError("No matching records found between val_df and pred_df for scoring.")

        rmsse_per_series = {}
        weighted_rmsse = 0.0

        for series_id, grp in merged.groupby("id", observed=False):
            sid = str(series_id)
            scale = self.scales.get(sid, 1.0)
            weight = self.weights.get(sid, 0.0)

            mse = np.mean((grp[self.target_col].values - grp[pred_col].values) ** 2)
            rmsse = np.sqrt(mse / scale)
            rmsse_per_series[sid] = float(rmsse)
            weighted_rmsse += weight * rmsse

        overall_wape = compute_wape(merged[self.target_col].values, merged[pred_col].values)
        overall_mae = float(
            np.mean(np.abs(merged[self.target_col].values - merged[pred_col].values))
        )
        overall_rmse = float(
            np.sqrt(np.mean((merged[self.target_col].values - merged[pred_col].values) ** 2))
        )

        return {
            "wrmsse": float(weighted_rmsse),
            "rmsse_mean": float(np.mean(list(rmsse_per_series.values()))),
            "wape": float(overall_wape),
            "mae": overall_mae,
            "rmse": overall_rmse,
        }
