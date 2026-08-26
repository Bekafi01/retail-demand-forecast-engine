"""Streamlit Page: Forecast Explorer with hierarchical slicing and conformal prediction intervals."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data.loader import generate_synthetic_m5_data
from src.data.preprocess import melt_sales_data, merge_calendar_and_prices
from src.evaluation.conformal import ConformalCalibrator
from src.features.pipeline import build_feature_table
from src.models.gbm import LightGBMForecaster

st.set_page_config(page_title="Forecast Explorer", page_icon="🔍", layout="wide")

st.title("🔍 Hierarchical Forecast Explorer & Uncertainty")
st.markdown(
    "Explore 28-day demand forecasts, actual historical sales, and calibrated **90% Conformal Prediction Intervals**."
)


@st.cache_data
def get_sample_data_and_forecasts():
    cal, prc, sal = generate_synthetic_m5_data(
        num_items=30, num_stores=3, num_days=180, random_seed=42
    )
    sales_long = melt_sales_data(sal)
    merged = merge_calendar_and_prices(sales_long, cal, prc)
    featured = build_feature_table(merged)

    dates = sorted(featured["date"].unique())
    train_df = featured[featured["date"] < dates[-28]]
    val_df = featured[featured["date"] >= dates[-28]]

    model = LightGBMForecaster(n_estimators=60, learning_rate=0.08)
    model.fit(train_df)
    preds = model.predict(val_df, horizon=28)

    conformal = ConformalCalibrator(normalized=True)
    conformal.fit(train_df["sales"].values[-len(preds) :], preds["y_pred"].values)
    intervals_df = conformal.predict_intervals(preds, alphas=[0.1, 0.2])

    merged_val = val_df.merge(
        intervals_df[["id", "date", "y_pred", "lower_90", "upper_90"]], on=["id", "date"]
    )
    return featured, merged_val


featured_df, forecast_df = get_sample_data_and_forecasts()

# Sidebar Filters
st.sidebar.header("Hierarchy Filters")
states = sorted(featured_df["state_id"].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", states)

stores = sorted(
    featured_df[featured_df["state_id"] == selected_state]["store_id"].unique().tolist()
)
selected_store = st.sidebar.selectbox("Select Store", stores)

cats = sorted(featured_df["cat_id"].unique().tolist())
selected_cat = st.sidebar.selectbox("Select Category", cats)

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

selected_item = st.sidebar.selectbox("Select Item Series", items)

# Plot Series Forecast
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
        line=dict(color="#1f2937", width=2.5),
    )
)

# Forecast Line
if not item_forecast.empty:
    fig.add_trace(
        go.Scatter(
            x=item_forecast["date"],
            y=item_forecast["y_pred"],
            mode="lines+markers",
            name="LightGBM Point Forecast",
            line=dict(color="#2563eb", width=2.5, dash="dash"),
        )
    )

    # Conformal 90% Bounds
    fig.add_trace(
        go.Scatter(
            x=item_forecast["date"].tolist() + item_forecast["date"].tolist()[::-1],
            y=item_forecast["upper_90"].tolist() + item_forecast["lower_90"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Conformal Interval",
            showlegend=True,
        )
    )

fig.update_layout(
    title=f"<b>Demand Trajectory & 28-Day Horizon Forecast</b> — <code>{selected_item}</code>",
    xaxis_title="Date",
    yaxis_title="Units Sold / Day",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

# Metrics table
col1, col2, col3 = st.columns(3)
if not item_forecast.empty:
    with col1:
        st.metric("Total Forecasted Units (28d)", f"{item_forecast['y_pred'].sum():.1f}")
    with col2:
        st.metric("Mean Daily Rate", f"{item_forecast['y_pred'].mean():.2f} units/day")
    with col3:
        wape = (item_forecast["sales"] - item_forecast["y_pred"]).abs().sum() / (
            item_forecast["sales"].sum() + 1e-4
        )
        st.metric("Validation Horizon WAPE", f"{wape * 100:.1f}%")
