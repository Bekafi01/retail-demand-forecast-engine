"""Unit tests for Population Stability Index (PSI) and data drift monitoring."""

import numpy as np
import pandas as pd

from mlops.monitoring.drift import DriftDetector, calculate_psi


def test_psi_identical_distributions():
    """Verify PSI is 0.0 (or nearly 0.0) for identical distributions."""
    np.random.seed(42)
    expected = np.random.normal(10.0, 2.0, 1000)
    actual = expected.copy()

    psi = calculate_psi(expected, actual)
    assert psi < 0.01


def test_psi_shifted_distributions():
    """Verify PSI exceeds critical threshold (>= 0.20) for heavily shifted distributions."""
    np.random.seed(42)
    expected = np.random.normal(10.0, 2.0, 1000)
    actual = np.random.normal(18.0, 2.0, 1000)

    psi = calculate_psi(expected, actual)
    assert psi >= 0.20


def test_drift_detector_report():
    """Verify DriftDetector generates structured report with alerts."""
    np.random.seed(42)
    base_df = pd.DataFrame(
        {
            "sell_price": np.random.uniform(2.0, 10.0, 500),
            "active_snap": np.random.choice([0, 1], 500),
        }
    )

    # Introduce drift in sell_price only
    curr_df = pd.DataFrame(
        {
            "sell_price": np.random.uniform(15.0, 30.0, 500),
            "active_snap": np.random.choice([0, 1], 500),
        }
    )

    detector = DriftDetector(psi_warning_threshold=0.10, psi_critical_threshold=0.20)
    detector.fit_baseline(base_df)
    report = detector.compute_drift_report(curr_df)

    assert "overall_status" in report
    assert "recommended_action" in report
    assert report["num_critical_features"] >= 1
    assert report["feature_metrics"]["sell_price"]["status"] == "CRITICAL_DRIFT"
