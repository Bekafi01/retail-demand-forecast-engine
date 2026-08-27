"""Streamlit Page: Forecast Explorer with hierarchical slicing and conformal prediction intervals."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils.demo_cache import get_or_create_demo_cache
from ui.styles import PLOTLY_LAYOUT, apply_custom_styles

st.set_page_config(page_title="Forecast Explorer", page_icon="🔍", layout="wide")
apply_custom_styles()

st.markdown(
    """
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
    <div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem;">
            <span class="badge badge-bronze">HIERARCHY SLICER</span>
            <span class="badge badge-sage">CONFORMAL CALIBRATED</span>
        </div>
        <h1 style="margin: 0; font-size: 2rem;">Demand Trajectory & Uncertainty Explorer</h1>
        <p style="color: #a89f91; margin: 0.2rem 0 0 0; font-size: 0.95rem;">
            Explore multi-level retail demand projections with distribution-free 90% Split Conformal Prediction intervals.
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_sample_data_and_forecasts():
    featured, forecast, _ = get_or_create_demo_cache(root_dir)
    return featured, forecast


featured_df, forecast_df = get_sample_data_and_forecasts()

# Hierarchy Slicing Sidebar
st.sidebar.markdown("### 🎛️ Hierarchy Slicing")
states = sorted(featured_df["state_id"].unique().tolist())
selected_state = st.sidebar.selectbox("State / Region", states)

stores = sorted(
    featured_df[featured_df["state_id"] == selected_state]["store_id"].unique().tolist()
)
selected_store = st.sidebar.selectbox("Store Location", stores)

cats = sorted(featured_df["cat_id"].unique().tolist())
selected_cat = st.sidebar.selectbox("Product Category", cats)

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

selected_item = st.sidebar.selectbox("Item Identifier (SKU)", items)

# Plot Forecast Series
item_history = featured_df[featured_df["id"] == selected_item].sort_values("date").tail(60)
item_forecast = forecast_df[forecast_df["id"] == selected_item].sort_values("date")

fig = go.Figure()

# Actual History
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

# Forecast & Uncertainty Ribbon
if not item_forecast.empty:
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

    # 90% Conformal Lower Bound & Fill Ribbon
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
    height=450,
)
fig.update_layout(**layout_opts)

st.plotly_chart(fig, use_container_width=True)

# Metric Summary Cards
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
