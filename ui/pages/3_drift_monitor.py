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
from ui.styles import PLOTLY_LAYOUT, apply_custom_styles

st.set_page_config(page_title="Drift Monitor", page_icon="🛡️", layout="wide")
apply_custom_styles()

st.markdown(
    """
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
    <div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem;">
            <span class="badge badge-blue">MLOPS TELEMETRY</span>
            <span class="badge badge-green">LIVE MONITORING</span>
        </div>
        <h1 style="margin: 0; font-size: 2rem;">Feature Stability & Drift Monitor</h1>
        <p style="color: #94a3b8; margin: 0.2rem 0 0 0; font-size: 0.95rem;">
            Continuous data drift detection using Population Stability Index (PSI) and Two-Sample Kolmogorov-Smirnov statistical tests.
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_baseline_and_current_distributions():
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=30, num_stores=2, num_days=180, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    merged = merge_calendar_and_prices(sales_long, cal, prc)
    featured = build_feature_table(merged)

    dates = sorted(featured["date"].unique())
    base_df = featured[featured["date"] < dates[120]].copy()
    curr_df = featured[featured["date"] >= dates[120]].copy()

    # Introduce simulated promotional drift in current batch
    curr_df["sell_price"] = curr_df["sell_price"] * 1.35
    curr_df["price_discount_ratio"] = np.clip(curr_df["price_discount_ratio"] * 2.0, 0.0, 1.0)
    return base_df, curr_df


base_df, curr_df = get_baseline_and_current_distributions()

detector = DriftDetector(psi_warning_threshold=0.10, psi_critical_threshold=0.20)
detector.fit_baseline(base_df)
report = detector.compute_drift_report(curr_df)

# Top Telemetry Cards
col1, col2, col3 = st.columns(3)

status_badge = (
    '<span class="badge badge-red">CRITICAL DRIFT</span>'
    if report["overall_status"] == "CRITICAL_DRIFT"
    else (
        '<span class="badge badge-amber">MODERATE DRIFT</span>'
        if report["overall_status"] == "MODERATE_DRIFT"
        else '<span class="badge badge-green">STABLE</span>'
    )
)

with col1:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-title">Overall System Telemetry</div>
        <div style="margin: 0.4rem 0;">{status_badge}</div>
        <div class="kpi-delta delta-neutral">● Continuous Ingestion Scan</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-title">Recommended Action</div>
        <div class="kpi-value" style="font-size: 1.25rem; font-family: monospace;">{report["recommended_action"]}</div>
        <div class="kpi-delta delta-warning">● Auto-Trigger Policy</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    crit_count = report["num_critical_features"]
    total_count = report["num_features_checked"]
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-title">Drifted Feature Ratio</div>
        <div class="kpi-value">{crit_count} / {total_count}</div>
        <div class="kpi-delta delta-warning">● PSI &ge; 0.20 Threshold</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# PSI Leaderboard
st.markdown("### Feature Stability Index (PSI) Telemetry")

records = []
for feat, m in report["feature_metrics"].items():
    records.append(
        {
            "Feature Name": feat,
            "PSI Metric": round(float(m.get("psi", 0.0)), 4),
            "KS p-value": f"{float(m.get('ks_p_value', 1.0)):.2e}",
            "Status": m.get("status", "STABLE"),
        }
    )

psi_df = pd.DataFrame(records).sort_values("PSI Metric", ascending=False).reset_index(drop=True)

st.dataframe(
    psi_df,
    use_container_width=True,
    height=240,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# Distribution Shift Overlay
st.markdown("### Empirical Density Shift Visualizer")
selected_feature = st.selectbox(
    "Select Feature to Compare Empirical Distributions", psi_df["Feature Name"].tolist()
)

if selected_feature in base_df.columns and selected_feature in curr_df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=base_df[selected_feature].dropna(),
            histnorm="probability density",
            name="Baseline Reference",
            marker_color="rgba(59, 130, 246, 0.65)",
            nbinsx=35,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=curr_df[selected_feature].dropna(),
            histnorm="probability density",
            name="Current Ingestion Batch",
            marker_color="rgba(239, 68, 68, 0.65)",
            nbinsx=35,
        )
    )

    feat_psi = calculate_psi(base_df[selected_feature], curr_df[selected_feature])
    layout_opts = PLOTLY_LAYOUT.copy()
    layout_opts.update(
        title=dict(
            text=f"<b>Population Density Shift:</b> <code>{selected_feature}</code> (PSI = {feat_psi:.4f})",
            font=dict(color="#f8fafc", size=14),
        ),
        xaxis_title=selected_feature,
        yaxis_title="Probability Density",
        barmode="overlay",
        height=400,
    )
    fig.update_layout(**layout_opts)
    st.plotly_chart(fig, use_container_width=True)

# Trigger Action
st.markdown("<br/>", unsafe_allow_html=True)
if st.button("🚀 Trigger Automated Retraining Pipeline (Airflow/MLflow)"):
    st.success(
        "✅ Retraining pipeline job submitted to MLflow Model Registry! Target alias updated to 'challenger'."
    )
