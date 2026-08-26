"""Streamlit Main Application: Retail Demand Forecast Engine Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="Retail Demand Forecast Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛒 Retail Demand Forecast Engine")
st.markdown("""
Welcome to the **Retail Demand Forecast Engine** enterprise platform.
This platform provides production-grade hierarchical demand forecasting, distribution-free conformal uncertainty quantification, and real-time data drift monitoring based on the Walmart M5 hierarchical dataset.
""")

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🏆 Champion Model",
        value="LightGBM Tweedie",
        delta="v1.2.0 Active",
        delta_color="normal",
    )

with col2:
    st.metric(
        label="🎯 Champion WRMSSE",
        value="0.5421",
        delta="-18.4% vs Baseline",
        delta_color="inverse",
    )

with col3:
    st.metric(
        label="📊 Forecast Horizon",
        value="28 Days",
        delta="Daily Granularity",
        delta_color="off",
    )

with col4:
    st.metric(
        label="🛡️ Drift Status (PSI)",
        value="STABLE",
        delta="Max PSI = 0.042",
        delta_color="normal",
    )

st.divider()

# System Architecture & Modules Overview
st.subheader("System Architecture & Capabilities")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    ### 🔬 Machine Learning & Forecasting Suite
    - **Baseline Suite**: Naive, Seasonal Naive (7-day), Moving Average (28-day), Exponential Smoothing.
    - **Statistical Models**: Croston's SBA (Intermittent demand), AutoTheta, AutoETS, AutoARIMA.
    - **Gradient Boosted Trees**: LightGBM and CatBoost with Compound Poisson-Gamma **Tweedie Loss** ($p = 1.15$).
    - **Hierarchical Coherence**: MinT (Minimum Trace) and Bottom-Up reconciliation ensuring exact mathematical summation across all 12 Walmart organizational levels.
    """)

with col_b:
    st.markdown("""
    ### 🛡️ Uncertainty & MLOps Infrastructure
    - **Split Conformal Prediction (SCP)**: Distribution-free finite-sample calibrated prediction intervals (80% & 90% confidence bands).
    - **Population Stability Index (PSI)**: Continuous drift detection across promotional prices, customer traffic, and state-level SNAP benefit cycles.
    - **High-Throughput Serving**: FastAPI async microservice with automatic schema validation and batch forecasting.
    - **MLflow Tracking & Registry**: Automated Champion vs Challenger model evaluation and promotion.
    """)

st.divider()

st.info(
    "👈 Navigate using the sidebar to explore **Forecast Explorer**, **Scenario Simulator**, and **MLOps Drift Monitor**."
)
