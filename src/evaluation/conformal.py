"""Split Conformal Prediction (SCP) for distribution-free retail uncertainty quantification."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConformalCalibrator:
    """Split Conformal Prediction Calibrator providing finite-sample coverage guarantees.

    Supports:
        - Standard Absolute Residuals Non-Conformity: R_i = |y_i - y_hat_i|
        - Normalized / Adaptive Non-Conformity: R_i = |y_i - y_hat_i| / (sigma_i + eps)
    """

    def __init__(
        self,
        normalized: bool = False,
        eps: float = 1e-3,
    ):
        self.normalized = normalized
        self.eps = eps
        self.cal_scores: Optional[np.ndarray] = None
        self.n_cal: int = 0
        self.is_calibrated: bool = False

    def fit(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_scale: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "ConformalCalibrator":
        """Compute calibration non-conformity scores on out-of-fold holdout predictions."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        if len(y_t) != len(y_p):
            raise ValueError("y_true and y_pred must have identical length.")

        residuals = np.abs(y_t - y_p)
        if self.normalized:
            if y_scale is None:
                scale = np.sqrt(np.maximum(y_p, 0.1)) + self.eps
            else:
                scale = np.asarray(y_scale, dtype=np.float64) + self.eps
            self.cal_scores = residuals / scale
        else:
            self.cal_scores = residuals

        self.n_cal = len(self.cal_scores)
        self.is_calibrated = True
        logger.info(f"Conformal calibrator fitted with {self.n_cal} calibration residuals.")
        return self

    def predict_intervals(
        self,
        df_pred: pd.DataFrame,
        alphas: Optional[List[float]] = None,
        pred_col: str = "y_pred",
        scale_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute lower and upper calibrated prediction bounds for given significance levels alphas.

        Args:
            df_pred: DataFrame with point forecast column `pred_col`
            alphas: List of miscoverage rates (e.g. [0.1, 0.2] for 90% and 80% intervals)
            pred_col: Column name containing point forecasts
            scale_col: Optional column name containing uncertainty scale

        Returns:
            DataFrame with added bound columns: `lower_{1-alpha}`, `upper_{1-alpha}`
        """
        if not self.is_calibrated or self.cal_scores is None:
            raise ValueError("Calibrator must be fitted before predict_intervals().")

        if alphas is None:
            alphas = [0.1, 0.2, 0.5]  # 90%, 80%, 50% intervals

        out = df_pred.copy()
        y_p = out[pred_col].values.astype(np.float64)

        if self.normalized:
            if scale_col and scale_col in out.columns:
                scale = out[scale_col].values.astype(np.float64) + self.eps
            else:
                scale = np.sqrt(np.maximum(y_p, 0.1)) + self.eps
        else:
            scale = np.ones_like(y_p)

        for alpha in alphas:
            conf_level = int(round((1 - alpha) * 100))
            q_level = min(1.0, np.ceil((self.n_cal + 1) * (1 - alpha)) / self.n_cal)
            q_val = np.quantile(self.cal_scores, q_level, method="higher")

            margin = q_val * scale
            out[f"lower_{conf_level}"] = np.maximum(0.0, y_p - margin).astype(np.float32)
            out[f"upper_{conf_level}"] = np.maximum(0.0, y_p + margin).astype(np.float32)

        return out

    @staticmethod
    def evaluate_coverage(
        y_true: Union[pd.Series, np.ndarray],
        lower: Union[pd.Series, np.ndarray],
        upper: Union[pd.Series, np.ndarray],
        alpha: float = 0.1,
    ) -> Dict[str, float]:
        """Compute empirical coverage rate, mean interval width, and Winkler Score."""
        y_t = np.asarray(y_true, dtype=np.float64)
        low_bound = np.asarray(lower, dtype=np.float64)
        up_bound = np.asarray(upper, dtype=np.float64)

        covered = (y_t >= low_bound) & (y_t <= up_bound)
        coverage_rate = float(np.mean(covered))
        mean_width = float(np.mean(up_bound - low_bound))

        # Winkler Score
        winkler_penalties = (
            (up_bound - low_bound)
            + ((2.0 / alpha) * (low_bound - y_t) * (y_t < low_bound))
            + ((2.0 / alpha) * (y_t - up_bound) * (y_t > up_bound))
        )
        mean_winkler = float(np.mean(winkler_penalties))

        return {
            "target_coverage": 1.0 - alpha,
            "empirical_coverage": coverage_rate,
            "mean_interval_width": mean_width,
            "mean_winkler_score": mean_winkler,
        }
