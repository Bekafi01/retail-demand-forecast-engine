"""Streamlit Page: What-If Pricing & Promotion Scenario Simulator."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils.demo_cache import get_or_create_demo_cache
from ui.styles import PLOTLY_LAYOUT, apply_custom_styles

st.set_page_config(page_title="Scenario Simulator", page_icon="🎮", layout="wide")
apply_custom_styles()

st.markdown(
    """
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
    <div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem;">
            <span class="badge badge-bronze">WHAT-IF ENGINE</span>
            <span class="badge badge-terracotta">REVENUE SIMULATION</span>
        </div>
        <h1 style="margin: 0; font-size: 2rem;">Promotional & SNAP Scenario Simulator</h1>
        <p style="color: #a89f91; margin: 0.2rem 0 0 0; font-size: 0.95rem;">
            Simulate the demand elasticity and revenue impact of promotional discounts and government SNAP benefit windows.
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_trained_simulator_model():
    featured, _, model = get_or_create_demo_cache(root_dir)
    return model, featured


model, base_features = get_trained_simulator_model()

# Controls
st.markdown("### Simulation Parameters")
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
    horizon = st.slider("Simulation Horizon (Days)", min_value=7, max_value=28, value=14, step=7)

sample_item = base_features["id"].iloc[0]
sim_df = base_features[base_features["id"] == sample_item].sort_values("date").tail(horizon).copy()

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
    height=400,
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
