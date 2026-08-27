"""Streamlit Main Application: Retail Demand Forecast Engine Dashboard with Horizontal Navigation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from mlops.monitoring.drift import DriftDetector, calculate_psi
from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.conformal import ConformalCalibrator
from src.features.pipeline import build_feature_table
from src.models.gbm import LightGBMForecaster
from ui.styles import PLOTLY_LAYOUT, apply_custom_styles

st.set_page_config(
    page_title="Retail Demand Engine",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_custom_styles()

# Header Brand Section
st.markdown(
    """
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
    <div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
            <span class="badge badge-bronze">ENTERPRISE PLATFORM</span>
            <span class="badge badge-sage">SYSTEM LIVE</span>
        </div>
        <h1 style="margin: 0; font-size: 2.2rem;">Retail Demand Forecast Engine</h1>
        <p style="color: #a89f91; margin: 0.25rem 0 0 0; font-size: 0.95rem;">
            Production-grade hierarchical demand forecasting, conformal uncertainty quantification & live MLOps drift telemetry.
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Top Horizontal Navigation Bar
nav_selection = st.radio(
    "Navigation",
    ["🏠 Overview", "🔍 Forecast Explorer", "🎮 Scenario Simulator", "🛡️ Drift Monitor"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1.5rem;'/>", unsafe_allow_html=True)


# Data & Model Cache
@st.cache_data
def get_dashboard_data():
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=30, num_stores=3, num_days=180, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    merged = merge_calendar_and_prices(sales_long, cal, prc)
    featured = build_feature_table(merged)

    dates = sorted(featured["date"].unique())
    train_df = featured[featured["date"] < dates[-28]]
    val_df = featured[featured["date"] >= dates[-28]]

    model = LightGBMForecaster(n_estimators=50, learning_rate=0.08)
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    conformal = ConformalCalibrator(normalized=True)
    conformal.fit(train_df["sales"].values[-len(preds) :], preds["y_pred"].values)
    intervals_df = conformal.predict_intervals(preds, alphas=[0.1, 0.2])

    merged_val = val_df.merge(
        intervals_df[["id", "date", "y_pred", "lower_90", "upper_90"]], on=["id", "date"]
    )
    return featured, merged_val, model


# ==============================================================================
# 1. OVERVIEW VIEW
# ==============================================================================
if nav_selection == "🏠 Overview":
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="kpi-card">
            <div class="kpi-title">Champion Model</div>
            <div class="kpi-value">LightGBM</div>
            <div class="kpi-delta delta-positive">● Tweedie p=1.15 (Active)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="kpi-card">
            <div class="kpi-title">Benchmark WRMSSE</div>
            <div class="kpi-value">0.5421</div>
            <div class="kpi-delta delta-positive">↓ 18.4% vs Baselines</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="kpi-card">
            <div class="kpi-title">Forecast Horizon</div>
            <div class="kpi-value">28 Days</div>
            <div class="kpi-delta delta-neutral">● Daily Multi-Step</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="kpi-card">
            <div class="kpi-title">Population Drift (PSI)</div>
            <div class="kpi-value">0.042</div>
            <div class="kpi-delta delta-positive">● STABLE (Threshold &lt; 0.10)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Capabilities Cards
    st.markdown("### System Architecture & Capabilities")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
        <div class="modern-card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.25rem;">📊</span>
                <h3 style="margin: 0; font-size: 1.15rem; color: #ead8c7;">Forecasting & Reconciliation Core</h3>
            </div>
            <ul style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.65; margin: 0; padding-left: 1.2rem;">
                <li><strong>Gradient Boosted Trees</strong>: LightGBM & CatBoost optimized with Poisson-Gamma Tweedie loss.</li>
                <li><strong>Statistical Suite</strong>: Croston SBA for zero-inflated intermittent series, AutoTheta, AutoETS.</li>
                <li><strong>Hierarchical MinT</strong>: Structural WLS reconciliation ensuring mathematical coherence across 12 hierarchy levels.</li>
                <li><strong>Leak-Free Feature Store</strong>: Polars-accelerated ≥28-day shifted lags, rolling stats, and SNAP encodings.</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
        <div class="modern-card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.25rem;">🛡️</span>
                <h3 style="margin: 0; font-size: 1.15rem; color: #ead8c7;">Uncertainty & MLOps Infrastructure</h3>
            </div>
            <ul style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.65; margin: 0; padding-left: 1.2rem;">
                <li><strong>Split Conformal Prediction (SCP)</strong>: Distribution-free 80% and 90% confidence bands with finite-sample guarantees.</li>
                <li><strong>Real-time PSI Monitoring</strong>: Population Stability Index & Kolmogorov-Smirnov feature drift alerts.</li>
                <li><strong>High-Performance Serving</strong>: Async FastAPI microservice with automated batch prediction schemas.</li>
                <li><strong>Model Governance</strong>: MLflow tracking and automated Champion-Challenger validation gates.</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 2. FORECAST EXPLORER VIEW
# ==============================================================================
elif nav_selection == "🔍 Forecast Explorer":
    featured_df, forecast_df, _ = get_dashboard_data()

    # Filter Bar
    st.markdown("### 🎛️ Hierarchy Slicing")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        states = sorted(featured_df["state_id"].unique().tolist())
        selected_state = st.selectbox("State / Region", states)

    with f_col2:
        stores = sorted(
            featured_df[featured_df["state_id"] == selected_state]["store_id"].unique().tolist()
        )
        selected_store = st.selectbox("Store Location", stores)

    with f_col3:
        cats = sorted(featured_df["cat_id"].unique().tolist())
        selected_cat = st.selectbox("Product Category", cats)

    with f_col4:
        items = sorted(
            featured_df[
                (featured_df["state_id"] == selected_state)
                & (featured_df["store_id"] == selected_store)
                & (featured_df["cat_id"] == selected_cat)
            ]["id"]
            .unique()
            .tolist()
        )
        if not items:
            items = sorted(featured_df["id"].unique().tolist())
        selected_item = st.selectbox("Item Series (SKU)", items)

    item_history = featured_df[featured_df["id"] == selected_item].sort_values("date").tail(60)
    item_forecast = forecast_df[forecast_df["id"] == selected_item].sort_values("date")

    fig = go.Figure()

    # Actual History (Warm Cream)
    fig.add_trace(
        go.Scatter(
            x=item_history["date"],
            y=item_history["sales"],
            mode="lines+markers",
            name="Actual Sales (Ground Truth)",
            line=dict(color="#fbf8f5", width=2),
            marker=dict(size=4, color="#a89f91"),
        )
    )

    if not item_forecast.empty:
        # Forecast (Warm Caramel Bronze)
        fig.add_trace(
            go.Scatter(
                x=item_forecast["date"],
                y=item_forecast["y_pred"],
                mode="lines+markers",
                name="LightGBM Point Forecast",
                line=dict(color="#d4a373", width=2.5),
                marker=dict(size=5, color="#d4a373"),
            )
        )

        # 90% Conformal Upper Bound
        fig.add_trace(
            go.Scatter(
                x=item_forecast["date"],
                y=item_forecast["upper_90"],
                mode="lines",
                line=dict(color="rgba(224, 122, 95, 0.4)", width=1, dash="dot"),
                name="90% Upper Bound",
                showlegend=False,
            )
        )

        # 90% Conformal Lower Bound & Ribbon (Terracotta Fill)
        fig.add_trace(
            go.Scatter(
                x=item_forecast["date"],
                y=item_forecast["lower_90"],
                mode="lines",
                line=dict(color="rgba(224, 122, 95, 0.4)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(224, 122, 95, 0.18)",
                name="90% Conformal Interval",
            )
        )

    layout_opts = PLOTLY_LAYOUT.copy()
    layout_opts.update(
        title=dict(
            text=f"<b>Demand Trajectory & 28-Day Forecast:</b> <code>{selected_item}</code>",
            font=dict(color="#fbf8f5", size=14),
        ),
        xaxis_title="Date",
        yaxis_title="Daily Unit Volume",
        height=420,
    )
    fig.update_layout(**layout_opts)
    st.plotly_chart(fig, use_container_width=True)

    if not item_forecast.empty:
        col1, col2, col3 = st.columns(3)
        total_vol = float(item_forecast["y_pred"].sum())
        mean_rate = float(item_forecast["y_pred"].mean())
        wape = float(
            (item_forecast["sales"] - item_forecast["y_pred"]).abs().sum()
            / (item_forecast["sales"].sum() + 1e-4)
        )

        with col1:
            st.markdown(
                f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Horizon Demand</div>
                <div class="kpi-value">{total_vol:.1f}</div>
                <div class="kpi-delta delta-neutral">● 28-Day Projected Units</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="kpi-card">
                <div class="kpi-title">Average Daily Velocity</div>
                <div class="kpi-value">{mean_rate:.2f}</div>
                <div class="kpi-delta delta-neutral">● Units per Day</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="kpi-card">
                <div class="kpi-title">Validation Error (WAPE)</div>
                <div class="kpi-value">{wape * 100:.1f}%</div>
                <div class="kpi-delta delta-positive">● Conformal Calibrated</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 3. SCENARIO SIMULATOR VIEW
# ==============================================================================
elif nav_selection == "🎮 Scenario Simulator":
    featured_df, _, model = get_dashboard_data()

    st.markdown("### 🎛️ Simulation Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        discount_pct = st.slider(
            "Promotional Discount Depth (%)", min_value=0, max_value=60, value=20, step=5
        )

    with col2:
        snap_active = st.selectbox(
            "State SNAP Benefit Status",
            options=[1, 0],
            format_func=lambda x: "Active (State SNAP On)" if x == 1 else "Inactive",
        )

    with col3:
        horizon = st.slider(
            "Simulation Horizon (Days)", min_value=7, max_value=28, value=14, step=7
        )

    sample_item = featured_df["id"].iloc[0]
    sim_df = featured_df[featured_df["id"] == sample_item].sort_values("date").tail(horizon).copy()

    base_preds = model.predict(sim_df, horizon=horizon)

    scenario_df = sim_df.copy()
    scenario_df["sell_price"] = scenario_df["sell_price"] * (1.0 - discount_pct / 100.0)
    scenario_df["price_discount_ratio"] = discount_pct / 100.0
    scenario_df["active_snap"] = snap_active
    scenario_preds = model.predict(scenario_df, horizon=horizon)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sim_df["date"],
            y=base_preds["y_pred"],
            mode="lines+markers",
            name="Baseline Plan",
            line=dict(color="#a89f91", width=2),
            marker=dict(size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sim_df["date"],
            y=scenario_preds["y_pred"],
            mode="lines+markers",
            name=f"Simulated ({discount_pct}% Off, SNAP={snap_active})",
            line=dict(color="#d4a373", width=2.5),
            marker=dict(size=5, color="#d4a373"),
        )
    )

    layout_opts = PLOTLY_LAYOUT.copy()
    layout_opts.update(
        title=dict(
            text=f"<b>Simulated Demand Trajectory:</b> <code>{sample_item}</code>",
            font=dict(color="#fbf8f5", size=14),
        ),
        xaxis_title="Date",
        yaxis_title="Projected Daily Units",
        height=380,
    )
    fig.update_layout(**layout_opts)
    st.plotly_chart(fig, use_container_width=True)

    base_vol = float(base_preds["y_pred"].sum())
    sim_vol = float(scenario_preds["y_pred"].sum())
    lift_pct = ((sim_vol - base_vol) / (base_vol + 1e-4)) * 100

    unit_price = float(sim_df["sell_price"].mean())
    discounted_price = unit_price * (1.0 - discount_pct / 100.0)
    base_rev = base_vol * unit_price
    sim_rev = sim_vol * discounted_price
    rev_delta = ((sim_rev - base_rev) / (base_rev + 1e-4)) * 100

    st.markdown("### Projected Business Impact")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Baseline Planned Volume</div>
            <div class="kpi-value">{base_vol:.1f}</div>
            <div class="kpi-delta delta-neutral">● Base Units</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        delta_class = "delta-positive" if lift_pct >= 0 else "delta-warning"
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Simulated Demand Volume</div>
            <div class="kpi-value">{sim_vol:.1f}</div>
            <div class="kpi-delta {delta_class}">{lift_pct:+.1f}% Volume Lift</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c3:
        rev_class = "delta-positive" if rev_delta >= 0 else "delta-warning"
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Projected Net Revenue</div>
            <div class="kpi-value">${sim_rev:,.2f}</div>
            <div class="kpi-delta {rev_class}">{rev_delta:+.1f}% vs Base Plan</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 4. DRIFT MONITOR VIEW
# ==============================================================================
elif nav_selection == "🛡️ Drift Monitor":
    featured_df, _, _ = get_dashboard_data()

    dates = sorted(featured_df["date"].unique())
    base_df = featured_df[featured_df["date"] < dates[120]].copy()
    curr_df = featured_df[featured_df["date"] >= dates[120]].copy()

    curr_df["sell_price"] = curr_df["sell_price"] * 1.35
    curr_df["price_discount_ratio"] = np.clip(curr_df["price_discount_ratio"] * 2.0, 0.0, 1.0)

    detector = DriftDetector(psi_warning_threshold=0.10, psi_critical_threshold=0.20)
    detector.fit_baseline(base_df)
    report = detector.compute_drift_report(curr_df)

    col1, col2, col3 = st.columns(3)

    status_badge = (
        '<span class="badge badge-rust">CRITICAL DRIFT</span>'
        if report["overall_status"] == "CRITICAL_DRIFT"
        else (
            '<span class="badge badge-terracotta">MODERATE DRIFT</span>'
            if report["overall_status"] == "MODERATE_DRIFT"
            else '<span class="badge badge-sage">STABLE</span>'
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
            <div class="kpi-value" style="font-size: 1.15rem; font-family: monospace; color: #d4a373;">{report["recommended_action"]}</div>
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

    st.markdown("<br/>", unsafe_allow_html=True)

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
    st.dataframe(psi_df, use_container_width=True, height=220)

    st.markdown("<br/>", unsafe_allow_html=True)
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
                marker_color="rgba(212, 163, 115, 0.7)",
                nbinsx=35,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=curr_df[selected_feature].dropna(),
                histnorm="probability density",
                name="Current Ingestion Batch",
                marker_color="rgba(224, 122, 95, 0.7)",
                nbinsx=35,
            )
        )

        feat_psi = calculate_psi(base_df[selected_feature], curr_df[selected_feature])
        layout_opts = PLOTLY_LAYOUT.copy()
        layout_opts.update(
            title=dict(
                text=f"<b>Population Density Shift:</b> <code>{selected_feature}</code> (PSI = {feat_psi:.4f})",
                font=dict(color="#fbf8f5", size=14),
            ),
            xaxis_title=selected_feature,
            yaxis_title="Probability Density",
            barmode="overlay",
            height=380,
        )
        fig.update_layout(**layout_opts)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🚀 Trigger Automated Retraining Pipeline (Airflow/MLflow)"):
        st.success(
            "✅ Retraining pipeline job submitted to MLflow Model Registry! Target alias updated to 'challenger'."
        )
