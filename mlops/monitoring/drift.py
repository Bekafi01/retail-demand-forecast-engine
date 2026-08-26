"""Feature and prediction drift detection using Population Stability Index (PSI) and statistical distance."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_psi(
    expected: Union[np.ndarray, pd.Series],
    actual: Union[np.ndarray, pd.Series],
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between baseline (expected) and current (actual) distributions.

    Formula:
        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

    Industry Interpretation:
        - PSI < 0.10: No significant drift / Stable
        - 0.10 <= PSI < 0.20: Moderate drift / Monitor
        - PSI >= 0.20: Significant drift / Trigger Model Retraining
    """
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)

    # Remove NaNs
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]

    if len(e) == 0 or len(a) == 0:
        return 0.0

    # Determine bin edges based on expected baseline quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(e, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Extend outer boundaries to -inf and +inf so all actual observations are captured in bins
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute bin counts
    e_counts, _ = np.histogram(e, bins=bin_edges)
    a_counts, _ = np.histogram(a, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    e_pct = (e_counts / len(e)) + eps
    a_pct = (a_counts / len(a)) + eps

    # Normalize after smoothing
    e_pct = e_pct / np.sum(e_pct)
    a_pct = a_pct / np.sum(a_pct)

    # PSI calculation
    psi_val = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(np.maximum(0.0, psi_val))


class DriftDetector:
    """Detects feature drift, prediction drift, and data quality degradation."""

    def __init__(
        self,
        psi_warning_threshold: float = 0.10,
        psi_critical_threshold: float = 0.20,
    ):
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_critical_threshold = psi_critical_threshold
        self.baseline_df: Optional[pd.DataFrame] = None
        self.feature_cols: Optional[List[str]] = None

    def fit_baseline(
        self, baseline_df: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> "DriftDetector":
        """Store baseline reference distribution."""
        self.baseline_df = baseline_df.copy()
        if feature_cols is None:
            exclude = ["id", "date", "d", "wm_yr_wk", "sales", "y_pred"]
            self.feature_cols = [c for c in baseline_df.columns if c not in exclude]
        else:
            self.feature_cols = feature_cols

        logger.info(
            f"Drift detector fitted with baseline of {len(baseline_df):,} rows across {len(self.feature_cols)} features."
        )
        return self

    def compute_drift_report(
        self,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive drift metrics across all features and target predictions."""
        if self.baseline_df is None or self.feature_cols is None:
            raise ValueError("DriftDetector must be fitted with fit_baseline() first.")

        check_features = features or self.feature_cols
        feature_reports = {}
        drifted_count = 0

        for col in check_features:
            if col not in self.baseline_df.columns or col not in current_df.columns:
                continue

            base_s = self.baseline_df[col]
            curr_s = current_df[col]

            if pd.api.types.is_numeric_dtype(base_s):
                psi = calculate_psi(base_s, curr_s)
                try:
                    ks_stat, p_val = stats.ks_2samp(base_s.dropna(), curr_s.dropna())
                    ks_p_val = float(p_val)
                except Exception:
                    ks_p_val = 1.0

                if psi >= self.psi_critical_threshold:
                    status = "CRITICAL_DRIFT"
                    drifted_count += 1
                elif psi >= self.psi_warning_threshold:
                    status = "MODERATE_DRIFT"
                else:
                    status = "STABLE"

                feature_reports[col] = {
                    "type": "numeric",
                    "psi": float(psi),
                    "ks_p_value": ks_p_val,
                    "status": status,
                }
            else:
                base_dist = base_s.value_counts(normalize=True)
                curr_dist = curr_s.value_counts(normalize=True)
                all_cats = set(base_dist.index).union(set(curr_dist.index))
                tvd = 0.5 * sum(
                    abs(base_dist.get(c, 0.0) - curr_dist.get(c, 0.0)) for c in all_cats
                )

                status = (
                    "CRITICAL_DRIFT"
                    if tvd >= 0.20
                    else ("MODERATE_DRIFT" if tvd >= 0.10 else "STABLE")
                )
                if status == "CRITICAL_DRIFT":
                    drifted_count += 1

                feature_reports[col] = {
                    "type": "categorical",
                    "total_variation_distance": float(tvd),
                    "status": status,
                }

        overall_status = (
            "CRITICAL_DRIFT"
            if drifted_count >= 1
            else ("MODERATE_DRIFT" if drifted_count > 0 else "STABLE")
        )
        action = (
            "TRIGGER_RETRAINING"
            if overall_status == "CRITICAL_DRIFT"
            else ("MONITOR" if overall_status == "MODERATE_DRIFT" else "NO_ACTION")
        )

        return {
            "overall_status": overall_status,
            "recommended_action": action,
            "num_features_checked": len(feature_reports),
            "num_critical_features": drifted_count,
            "feature_metrics": feature_reports,
        }
