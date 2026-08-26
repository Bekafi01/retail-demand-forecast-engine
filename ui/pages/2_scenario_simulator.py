"""Streamlit Page: What-If Pricing & Promotion Scenario Simulator."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.features.pipeline import build_feature_table
from src.models.gbm import LightGBMForecaster

st.set_page_config(page_title="Scenario Simulator", page_icon="🎮", layout="wide")

st.title("🎮 What-If Pricing & Promotion Scenario Simulator")
st.markdown("""
Simulate the business impact of **price discounts (% discount)** and **SNAP benefit disbursement windows** on forecasted demand.
""")


@st.cache_resource
def get_trained_simulator_model():
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=25, num_stores=2, num_days=120, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    merged = merge_calendar_and_prices(sales_long, cal, prc)
    featured = build_feature_table(merged)

    model = LightGBMForecaster(n_estimators=50, learning_rate=0.08)
    model.fit(featured)
    return model, featured


model, base_features = get_trained_simulator_model()

# Controls
st.subheader("Simulation Controls")
col1, col2, col3 = st.columns(3)

with col1:
    discount_pct = st.slider(
        "💰 Promotional Discount Depth (%)", min_value=0, max_value=60, value=20, step=5
    )

with col2:
    snap_active = st.selectbox(
        "🍎 SNAP Benefit Active Status",
        options=[1, 0],
        format_func=lambda x: "Active (State SNAP On)" if x == 1 else "Inactive",
    )

with col3:
    horizon = st.slider("📅 Simulation Horizon (Days)", min_value=7, max_value=28, value=14, step=7)

# Run Simulation
sample_item = base_features["id"].iloc[0]
sim_df = base_features[base_features["id"] == sample_item].sort_values("date").tail(horizon).copy()

# Baseline Prediction
base_preds = model.predict(sim_df, horizon=horizon)

# Scenario Prediction (modified price & discount)
scenario_df = sim_df.copy()
scenario_df["sell_price"] = scenario_df["sell_price"] * (1.0 - discount_pct / 100.0)
scenario_df["price_discount_ratio"] = discount_pct / 100.0
scenario_df["active_snap"] = snap_active
scenario_preds = model.predict(scenario_df, horizon=horizon)

# Visual Comparison
comp_df = pd.DataFrame(
    {
        "date": sim_df["date"],
        "Baseline Forecast": base_preds["y_pred"].values,
        f"Simulated ({discount_pct}% Off, SNAP={snap_active})": scenario_preds["y_pred"].values,
    }
)

fig = px.line(
    comp_df,
    x="date",
    y=["Baseline Forecast", f"Simulated ({discount_pct}% Off, SNAP={snap_active})"],
    title=f"<b>Scenario Impact on Demand Forecast:</b> {sample_item}",
    color_discrete_sequence=["#64748b", "#059669"],
    markers=True,
    template="plotly_white",
)
fig.update_layout(xaxis_title="Date", yaxis_title="Predicted Daily Units", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Business Impact Summary
base_vol = float(base_preds["y_pred"].sum())
sim_vol = float(scenario_preds["y_pred"].sum())
lift_pct = ((sim_vol - base_vol) / (base_vol + 1e-4)) * 100

st.subheader("Business Impact Summary")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Baseline Volume", f"{base_vol:.1f} units")
with c2:
    st.metric("Simulated Volume", f"{sim_vol:.1f} units", delta=f"{lift_pct:+.1f}% Lift")
with c3:
    unit_price = float(sim_df["sell_price"].mean())
    discounted_price = unit_price * (1.0 - discount_pct / 100.0)
    base_rev = base_vol * unit_price
    sim_rev = sim_vol * discounted_price
    rev_delta = ((sim_rev - base_rev) / (base_rev + 1e-4)) * 100
    st.metric("Projected Revenue", f"${sim_rev:,.2f}", delta=f"{rev_delta:+.1f}% vs Base")
