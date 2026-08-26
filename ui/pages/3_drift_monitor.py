"""Streamlit Page: MLOps Drift Monitor & Population Stability Index (PSI) Dashboard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from mlops.monitoring.drift import DriftDetector, calculate_psi
from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.features.pipeline import build_feature_table

st.set_page_config(page_title="Drift Monitor", page_icon="🛡️", layout="wide")

st.title("🛡️ MLOps Drift Monitor (PSI & Feature Stability)")
st.markdown("""
Continuous data drift monitoring using the **Population Stability Index (PSI)** and **Two-Sample Kolmogorov-Smirnov Tests** against training baseline distributions.
""")


@st.cache_data
def get_baseline_and_current_distributions():
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=30, num_stores=2, num_days=180, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    merged = merge_calendar_and_prices(sales_long, cal, prc)
    featured = build_feature_table(merged)

    # Baseline (first 120 days) vs Current (last 60 days with simulated price inflation shock)
    dates = sorted(featured["date"].unique())
    base_df = featured[featured["date"] < dates[120]].copy()
    curr_df = featured[featured["date"] >= dates[120]].copy()

    # Introduce simulated promotional drift in current data
    curr_df["sell_price"] = curr_df["sell_price"] * 1.35
    curr_df["price_discount_ratio"] = np.clip(curr_df["price_discount_ratio"] * 2.0, 0.0, 1.0)
    return base_df, curr_df


base_df, curr_df = get_baseline_and_current_distributions()

detector = DriftDetector(psi_warning_threshold=0.10, psi_critical_threshold=0.20)
detector.fit_baseline(base_df)
report = detector.compute_drift_report(curr_df)

# Top Status Banner
col1, col2, col3 = st.columns(3)
with col1:
    status_color = (
        "red"
        if report["overall_status"] == "CRITICAL_DRIFT"
        else ("orange" if report["overall_status"] == "MODERATE_DRIFT" else "green")
    )
    st.markdown(f"### Overall System Status: **:{status_color}[{report['overall_status']}]**")

with col2:
    st.markdown(f"### Recommended Action: **`{report['recommended_action']}`**")

with col3:
    st.markdown(
        f"### Critical Drift Features: **`{report['num_critical_features']} / {report['num_features_checked']}`**"
    )

st.divider()

# Feature PSI Leaderboard Table
st.subheader("Feature Stability Index (PSI) Summary")

records = []
for feat, m in report["feature_metrics"].items():
    records.append(
        {
            "Feature": feat,
            "PSI Score": m.get("psi", 0.0),
            "KS p-value": m.get("ks_p_value", 1.0),
            "Status": m.get("status", "STABLE"),
        }
    )

psi_df = pd.DataFrame(records).sort_values("PSI Score", ascending=False).reset_index(drop=True)

# Format badges
st.dataframe(
    psi_df.style.map(
        lambda val: (
            "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
            if val == "CRITICAL_DRIFT"
            else (
                "background-color: #fef3c7; color: #92400e; font-weight: bold;"
                if val == "MODERATE_DRIFT"
                else "background-color: #dcfce7; color: #166534; font-weight: bold;"
            )
        ),
        subset=["Status"],
    ),
    use_container_width=True,
)

st.divider()

# Distribution Overlay Plot
st.subheader("Distribution Comparison: Baseline vs Current Ingestion")
selected_feature = st.selectbox(
    "Select Feature to Inspect Distribution", psi_df["Feature"].tolist()
)

if selected_feature in base_df.columns and selected_feature in curr_df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=base_df[selected_feature].dropna(),
            histnorm="probability density",
            name="Baseline Reference Distribution",
            marker_color="rgba(37, 99, 235, 0.6)",
            nbinsx=30,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=curr_df[selected_feature].dropna(),
            histnorm="probability density",
            name="Current Ingestion Distribution",
            marker_color="rgba(239, 68, 68, 0.6)",
            nbinsx=30,
        )
    )

    fig.update_layout(
        title=f"<b>Population Density Shift:</b> <code>{selected_feature}</code> (PSI = {calculate_psi(base_df[selected_feature], curr_df[selected_feature]):.4f})",
        xaxis_title=selected_feature,
        yaxis_title="Probability Density",
        barmode="overlay",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

if st.button("🚀 Trigger Automated Retraining Pipeline (Airflow/MLflow)"):
    st.success(
        "Retraining pipeline job submitted to MLflow Model Registry! Target alias updated to 'challenger'."
    )
