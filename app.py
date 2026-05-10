import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Retail Future Intelligence App",
    layout="wide"
)

st.title("Smart Retail Future Intelligence App")

st.markdown("""
This application uses backend PCA forecasting and anomaly intelligence to produce future-facing business outputs.

Visible business tabs:
- Profit Impact Simulator
- Business Action Recommendations
- Revenue Leakage Detection
- Inventory Optimization Engine
- Executive AI Summary

Select a future prediction date from the sidebar, and all KPIs, tables and charts will update for that future date.
""")

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# Detect Key Columns
# --------------------------------------------------
category_col = None
sku_col = None
date_col = None

for col in df.columns:
    lower_col = col.lower()

    if category_col is None and "category" in lower_col:
        category_col = col

    if sku_col is None and (
        "sku" in lower_col
        or "item" in lower_col
        or "product" in lower_col
        or "stockcode" in lower_col
    ):
        sku_col = col

    if date_col is None and "date" in lower_col:
        date_col = col

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("Filters")

filtered_df = df.copy()

if category_col:
    category_options = ["All"] + sorted(filtered_df[category_col].dropna().astype(str).unique())
    selected_category = st.sidebar.selectbox("Select Category", category_options)

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df[category_col].astype(str) == selected_category]

if sku_col:
    sku_options = ["All"] + sorted(filtered_df[sku_col].dropna().astype(str).unique())
    selected_sku = st.sidebar.selectbox("Select SKU / Item / Product", sku_options)

    if selected_sku != "All":
        filtered_df = filtered_df[filtered_df[sku_col].astype(str) == selected_sku]

if filtered_df.empty:
    st.error("No records available for the selected filters.")
    st.stop()

st.sidebar.success(f"Filtered records: {len(filtered_df)}")

# --------------------------------------------------
# Numeric Columns
# --------------------------------------------------
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 3:
    st.error("Dataset must contain at least 3 numeric columns.")
    st.stop()

# --------------------------------------------------
# Automatic Column Mapping
# --------------------------------------------------
def get_default_column(possible_names, exclude_words=None):
    exclude_words = exclude_words or []

    for name in possible_names:
        for col in numeric_cols:
            lower_col = col.lower()

            if any(ex_word in lower_col for ex_word in exclude_words):
                continue

            if name.lower() in lower_col:
                return col

    return None


quantity_col = get_default_column(["sales_qty", "quantity", "qty"])
revenue_col = get_default_column(["sales_revenue", "revenue", "amount", "sales"])
cost_col = get_default_column(["unit_cost", "cost"])
price_col = get_default_column(["unit_price", "price"])

stock_col = get_default_column(
    ["stock_on_hand", "inventory", "soh", "stock_qty", "stock_level"],
    exclude_words=["flag", "stockout"]
)

lead_time_col = get_default_column(["lead_time_days", "lead_time", "lead time"])
service_col = get_default_column(["delivery_reliability", "reliability", "service"])

# --------------------------------------------------
# Business Assumptions
# --------------------------------------------------
st.sidebar.header("Business Assumptions")

future_steps = 365

threshold_percentile = st.sidebar.slider(
    "Anomaly Threshold Percentile",
    90,
    99,
    95
)

expected_uplift_pct = st.sidebar.slider(
    "Expected Sales Uplift %",
    1,
    100,
    10
)

target_margin_pct = st.sidebar.slider(
    "Target Margin %",
    1,
    90,
    25
)

holding_cost_pct = st.sidebar.slider(
    "Inventory Holding Cost %",
    1,
    50,
    10
)

# --------------------------------------------------
# Backend Feature Selection - Hidden
# --------------------------------------------------
features = [
    col for col in [
        quantity_col,
        revenue_col,
        cost_col,
        price_col,
        stock_col,
        lead_time_col,
        service_col
    ]
    if col is not None and col in numeric_cols
]

if len(features) < 3:
    features = numeric_cols[:5]

# --------------------------------------------------
# Backend PCA + Forecasting
# --------------------------------------------------
X = filtered_df[features].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

if len(X) < 10:
    st.error("Not enough records after filtering. Please select broader filters.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = min(3, len(features))
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(X_scaled)

pc_names = [f"PC{i+1}" for i in range(n_components)]
pc_df = pd.DataFrame(pcs, columns=pc_names)

pc_forecasts = {}

for col in pc_df.columns:
    try:
        model = ARIMA(pc_df[col], order=(2, 1, 2))
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=future_steps)
        pc_forecasts[col] = forecast_result.predicted_mean.values
    except Exception:
        pc_forecasts[col] = np.repeat(pc_df[col].iloc[-1], future_steps)

pc_forecast_df = pd.DataFrame(pc_forecasts)

future_scaled = pca.inverse_transform(pc_forecast_df)
future_values = scaler.inverse_transform(future_scaled)
future_df = pd.DataFrame(future_values, columns=features)

# --------------------------------------------------
# Backend Anomaly Classification
# --------------------------------------------------
X_hat = pca.inverse_transform(pcs)
residual = X_scaled - X_hat

spe = np.sum(residual ** 2, axis=1)
eigen_vals = pca.explained_variance_
t2 = np.sum((pcs ** 2) / eigen_vals, axis=1)

spe_norm = (spe - np.min(spe)) / (np.max(spe) - np.min(spe) + 1e-9)
t2_norm = (t2 - np.min(t2)) / (np.max(t2) - np.min(t2) + 1e-9)
g2 = 0.5 * spe_norm + 0.5 * t2_norm

spe_threshold = np.percentile(spe, threshold_percentile)
t2_threshold = np.percentile(t2, threshold_percentile)
g2_threshold = np.percentile(g2, threshold_percentile)

anomaly = (
    (spe > spe_threshold)
    | (t2 > t2_threshold)
    | (g2 > g2_threshold)
)

results = pd.DataFrame({
    "SPE": spe,
    "T2": t2,
    "G2": g2,
    "Anomaly": anomaly.astype(int)
})

future_pcs = pc_forecast_df.values
future_scaled_reconstructed = pca.inverse_transform(future_pcs)
future_residual = future_scaled - future_scaled_reconstructed

future_spe = np.sum(future_residual ** 2, axis=1)
future_t2 = np.sum((future_pcs ** 2) / eigen_vals, axis=1)

future_spe_norm = (future_spe - np.min(spe)) / (np.max(spe) - np.min(spe) + 1e-9)
future_t2_norm = (future_t2 - np.min(t2)) / (np.max(t2) - np.min(t2) + 1e-9)
future_g2 = 0.5 * future_spe_norm + 0.5 * future_t2_norm

future_anomaly = (
    (future_spe > spe_threshold)
    | (future_t2 > t2_threshold)
    | (future_g2 > g2_threshold)
)

future_anomaly_df = pd.DataFrame({
    "Future_SPE": future_spe,
    "Future_T2": future_t2,
    "Future_G2": future_g2,
    "Future_Anomaly": future_anomaly.astype(int)
})

# --------------------------------------------------
# Future Date Selection - Today to 1 Year
# --------------------------------------------------
st.sidebar.header("Future Prediction Selection")

today = pd.Timestamp.today().date()
one_year_from_today = (pd.Timestamp.today() + pd.DateOffset(years=1)).date()

selected_future_date = st.sidebar.date_input(
    "Select Future Prediction Date",
    value=today,
    min_value=today,
    max_value=one_year_from_today
)

future_prediction_label = str(selected_future_date)
selected_date_text = future_prediction_label

days_ahead = (pd.to_datetime(selected_future_date).date() - today).days

if days_ahead <= 0:
    selected_forecast_index = 0
else:
    selected_forecast_index = min(days_ahead - 1, len(future_df) - 1)

selected_future_row = future_df.iloc[[selected_forecast_index]]
selected_future_anomaly = future_anomaly_df.iloc[[selected_forecast_index]]

selected_future_values = selected_future_row.iloc[0]

future_anomaly_flag = 0

if not selected_future_anomaly.empty:
    future_anomaly_flag = int(selected_future_anomaly["Future_Anomaly"].iloc[0])

# --------------------------------------------------
# Business Data Preparation
# --------------------------------------------------
business_df = filtered_df.copy()

if sku_col is None:
    business_df["SKU"] = "Unknown SKU"
    sku_col = "SKU"

if category_col is None:
    business_df["Category"] = "Unknown Category"
    category_col = "Category"

if quantity_col:
    business_df["Business_Quantity"] = pd.to_numeric(business_df[quantity_col], errors="coerce").fillna(0)
else:
    business_df["Business_Quantity"] = 1

if revenue_col:
    business_df["Business_Revenue"] = pd.to_numeric(business_df[revenue_col], errors="coerce").fillna(0)
elif price_col:
    business_df["Business_Revenue"] = (
        pd.to_numeric(business_df[price_col], errors="coerce").fillna(0)
        * business_df["Business_Quantity"]
    )
else:
    business_df["Business_Revenue"] = 0

if cost_col:
    business_df["Business_Cost"] = (
        pd.to_numeric(business_df[cost_col], errors="coerce").fillna(0)
        * business_df["Business_Quantity"]
    )
else:
    business_df["Business_Cost"] = business_df["Business_Revenue"] * 0.65

if stock_col:
    business_df["Business_Stock"] = pd.to_numeric(business_df[stock_col], errors="coerce").fillna(0)
else:
    business_df["Business_Stock"] = business_df["Business_Quantity"] * 2

if lead_time_col:
    business_df["Business_Lead_Time"] = pd.to_numeric(business_df[lead_time_col], errors="coerce").fillna(0)
else:
    business_df["Business_Lead_Time"] = 7

if service_col:
    business_df["Business_Service_Level"] = pd.to_numeric(business_df[service_col], errors="coerce").fillna(0)
else:
    business_df["Business_Service_Level"] = 95

business_df["Business_Profit"] = business_df["Business_Revenue"] - business_df["Business_Cost"]

business_df["Business_Margin_%"] = np.where(
    business_df["Business_Revenue"] > 0,
    business_df["Business_Profit"] / business_df["Business_Revenue"] * 100,
    0
)

# --------------------------------------------------
# Historical SKU Summary
# --------------------------------------------------
sku_summary = (
    business_df
    .groupby([category_col, sku_col], dropna=False)
    .agg(
        Total_Quantity=("Business_Quantity", "sum"),
        Total_Revenue=("Business_Revenue", "sum"),
        Total_Cost=("Business_Cost", "sum"),
        Total_Profit=("Business_Profit", "sum"),
        Avg_Margin_Percent=("Business_Margin_%", "mean"),
        Avg_Stock=("Business_Stock", "mean"),
        Avg_Lead_Time=("Business_Lead_Time", "mean"),
        Avg_Service_Level=("Business_Service_Level", "mean"),
        Record_Count=("Business_Revenue", "count")
    )
    .reset_index()
)

sku_summary["Revenue_Per_Unit"] = np.where(
    sku_summary["Total_Quantity"] > 0,
    sku_summary["Total_Revenue"] / sku_summary["Total_Quantity"],
    0
)

sku_summary["Profit_Per_Unit"] = np.where(
    sku_summary["Total_Quantity"] > 0,
    sku_summary["Total_Profit"] / sku_summary["Total_Quantity"],
    0
)

# --------------------------------------------------
# Future Factor Calculation
# --------------------------------------------------
future_revenue_factor = 1
future_quantity_factor = 1
future_cost_factor = 1
future_stock_factor = 1

if revenue_col and revenue_col in selected_future_values.index:
    hist_mean = business_df["Business_Revenue"].mean()
    if hist_mean != 0:
        future_revenue_factor = selected_future_values[revenue_col] / hist_mean

if quantity_col and quantity_col in selected_future_values.index:
    hist_mean = business_df["Business_Quantity"].mean()
    if hist_mean != 0:
        future_quantity_factor = selected_future_values[quantity_col] / hist_mean

if cost_col and cost_col in selected_future_values.index:
    hist_mean = business_df["Business_Cost"].mean()
    if hist_mean != 0:
        future_cost_factor = selected_future_values[cost_col] / hist_mean

if stock_col and stock_col in selected_future_values.index:
    hist_mean = business_df["Business_Stock"].mean()
    if hist_mean != 0:
        future_stock_factor = selected_future_values[stock_col] / hist_mean

sku_summary["Future_Predicted_Quantity"] = sku_summary["Total_Quantity"] * future_quantity_factor
sku_summary["Future_Predicted_Revenue"] = sku_summary["Total_Revenue"] * future_revenue_factor
sku_summary["Future_Predicted_Cost"] = sku_summary["Total_Cost"] * future_cost_factor

sku_summary["Future_Predicted_Profit"] = (
    sku_summary["Future_Predicted_Revenue"]
    - sku_summary["Future_Predicted_Cost"]
)

sku_summary["Future_Predicted_Margin_%"] = np.where(
    sku_summary["Future_Predicted_Revenue"] > 0,
    sku_summary["Future_Predicted_Profit"] / sku_summary["Future_Predicted_Revenue"] * 100,
    0
)

sku_summary["Future_Predicted_Stock"] = sku_summary["Avg_Stock"] * future_stock_factor

sku_summary["Future_Uplift_Revenue"] = (
    sku_summary["Future_Predicted_Revenue"] * expected_uplift_pct / 100
)

sku_summary["Future_Uplift_Profit"] = (
    sku_summary["Future_Predicted_Profit"] * expected_uplift_pct / 100
)

# --------------------------------------------------
# Future Revenue Leakage Logic
# --------------------------------------------------
future_revenue_threshold = sku_summary["Future_Predicted_Revenue"].median()
future_stock_threshold = sku_summary["Future_Predicted_Stock"].median()

sku_summary["Future_Revenue_Leakage_Flag"] = np.where(
    (
        (sku_summary["Future_Predicted_Revenue"] < future_revenue_threshold)
        | (sku_summary["Future_Predicted_Margin_%"] < target_margin_pct)
        | (sku_summary["Future_Predicted_Stock"] > future_stock_threshold)
    ),
    "Future Leakage Risk",
    "Healthy"
)

sku_summary["Future_Revenue_Leakage_Reason"] = np.select(
    [
        sku_summary["Future_Predicted_Margin_%"] < target_margin_pct,
        sku_summary["Future_Predicted_Revenue"] < future_revenue_threshold,
        sku_summary["Future_Predicted_Stock"] > future_stock_threshold
    ],
    [
        "Future margin is below target",
        "Future revenue is below median",
        "Future stock may be excessive"
    ],
    default="No major future leakage signal"
)

sku_summary["Future_Estimated_Revenue_Leakage"] = np.where(
    sku_summary["Future_Revenue_Leakage_Flag"] == "Future Leakage Risk",
    np.maximum(
        sku_summary["Future_Predicted_Revenue"] * 0.10,
        sku_summary["Future_Predicted_Stock"] * sku_summary["Revenue_Per_Unit"] * 0.05
    ),
    0
)

# --------------------------------------------------
# Future Inventory Optimisation Logic
# --------------------------------------------------
sku_summary["Estimated_Demand_Next_Period"] = (
    sku_summary["Future_Predicted_Quantity"]
    / sku_summary["Record_Count"].replace(0, 1)
)

sku_summary["Recommended_Safety_Stock"] = (
    sku_summary["Estimated_Demand_Next_Period"]
    * (sku_summary["Avg_Lead_Time"] / 7)
    * 0.5
)

sku_summary["Recommended_Reorder_Point"] = (
    sku_summary["Estimated_Demand_Next_Period"]
    * (sku_summary["Avg_Lead_Time"] / 7)
) + sku_summary["Recommended_Safety_Stock"]

sku_summary["Future_Inventory_Status"] = np.select(
    [
        sku_summary["Future_Predicted_Stock"] < sku_summary["Recommended_Reorder_Point"],
        sku_summary["Future_Predicted_Stock"] > sku_summary["Recommended_Reorder_Point"] * 2
    ],
    [
        "Future Reorder Required",
        "Future Possible Overstock"
    ],
    default="Future Balanced"
)

sku_summary["Future_Inventory_Action"] = np.select(
    [
        sku_summary["Future_Inventory_Status"] == "Future Reorder Required",
        sku_summary["Future_Inventory_Status"] == "Future Possible Overstock"
    ],
    [
        "Increase replenishment before selected future period",
        "Reduce buying or run promotion before selected future period"
    ],
    default="Maintain planned inventory level"
)

sku_summary["Future_Estimated_Holding_Cost"] = (
    sku_summary["Future_Predicted_Stock"]
    * sku_summary["Revenue_Per_Unit"]
    * holding_cost_pct / 100
)

# --------------------------------------------------
# Future Business Action Logic
# --------------------------------------------------
sku_summary["Future_Business_Action"] = np.select(
    [
        (
            (sku_summary["Future_Predicted_Revenue"] >= future_revenue_threshold)
            & (sku_summary["Future_Predicted_Margin_%"] >= target_margin_pct)
        ),
        (
            (sku_summary["Future_Predicted_Revenue"] >= future_revenue_threshold)
            & (sku_summary["Future_Predicted_Margin_%"] < target_margin_pct)
        ),
        sku_summary["Future_Inventory_Status"] == "Future Possible Overstock",
        sku_summary["Future_Inventory_Status"] == "Future Reorder Required"
    ],
    [
        "Promote / Prioritise SKU",
        "Review pricing or cost",
        "Clear excess stock with promotion",
        "Replenish inventory"
    ],
    default="Monitor"
)

sku_summary["Future_Action_Priority"] = np.select(
    [
        sku_summary["Future_Business_Action"].isin(["Promote / Prioritise SKU", "Replenish inventory"]),
        sku_summary["Future_Business_Action"].isin(["Review pricing or cost", "Clear excess stock with promotion"])
    ],
    ["High", "Medium"],
    default="Low"
)

# --------------------------------------------------
# Visible Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Profit Impact Simulator",
    "2. Business Action Recommendations",
    "3. Revenue Leakage Detection",
    "4. Inventory Optimization Engine",
    "5. Executive AI Summary"
])

# ==================================================
# TAB 1 — Profit Impact Simulator
# ==================================================
with tab1:
    st.header("Profit Impact Simulator")
    st.info(f"Showing prediction-based output for: {selected_date_text}")

    st.markdown(
        f"**Business Impact:** Shows the predicted revenue and profit opportunity for the selected future date: **{selected_date_text}**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Predicted Revenue on {selected_date_text}", f"{sku_summary['Future_Predicted_Revenue'].sum():,.2f}")
    c2.metric(f"Predicted Profit on {selected_date_text}", f"{sku_summary['Future_Predicted_Profit'].sum():,.2f}")
    c3.metric(f"Predicted Margin % on {selected_date_text}", f"{sku_summary['Future_Predicted_Margin_%'].mean():.2f}%")
    c4.metric(f"Anomaly Risk on {selected_date_text}", "Yes" if future_anomaly_flag == 1 else "No")

    profit_df = sku_summary.sort_values("Future_Predicted_Profit", ascending=False)

    profit_cols = [
        category_col,
        sku_col,
        "Total_Revenue",
        "Total_Profit",
        "Future_Predicted_Revenue",
        "Future_Predicted_Profit",
        "Future_Predicted_Margin_%",
        "Future_Uplift_Revenue",
        "Future_Uplift_Profit"
    ]

    st.subheader(f"Future Profit Impact Table for {selected_date_text}")
    st.dataframe(profit_df[profit_cols], use_container_width=True)

    top_10 = profit_df.head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_10[sku_col].astype(str),
        y=top_10["Future_Predicted_Profit"],
        name=f"Predicted Profit on {selected_date_text}"
    ))

    fig.add_trace(go.Bar(
        x=top_10[sku_col].astype(str),
        y=top_10["Future_Uplift_Profit"],
        name=f"Uplift Profit on {selected_date_text}"
    ))

    fig.update_layout(
        title=f"Top 10 SKUs: Future Profit Impact for {selected_date_text}",
        xaxis_title="SKU",
        yaxis_title="Profit",
        barmode="group",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 2 — Business Action Recommendations
# ==================================================
with tab2:
    st.header("Business Action Recommendations")
    st.info(f"Showing prediction-based output for: {selected_date_text}")

    st.markdown(
        f"**Business Impact:** Converts the prediction for **{selected_date_text}** into clear SKU-level actions such as promote, replenish, reprice, clear stock, or monitor."
    )

    action_df = sku_summary.sort_values(
        ["Future_Action_Priority", "Future_Predicted_Profit"],
        ascending=[True, False]
    )

    action_cols = [
        category_col,
        sku_col,
        "Future_Predicted_Revenue",
        "Future_Predicted_Profit",
        "Future_Predicted_Margin_%",
        "Future_Predicted_Stock",
        "Future_Inventory_Status",
        "Future_Revenue_Leakage_Flag",
        "Future_Business_Action",
        "Future_Action_Priority"
    ]

    st.subheader(f"Future Business Action Table for {selected_date_text}")
    st.dataframe(action_df[action_cols], use_container_width=True)

    action_summary = (
        action_df
        .groupby(["Future_Business_Action", "Future_Action_Priority"])
        .size()
        .reset_index(name="SKU_Count")
        .sort_values("SKU_Count", ascending=False)
    )

    st.subheader(f"Future Action Summary for {selected_date_text}")
    st.dataframe(action_summary, use_container_width=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=action_summary["Future_Business_Action"],
        y=action_summary["SKU_Count"],
        name=f"SKU Count on {selected_date_text}"
    ))

    fig.update_layout(
        title=f"Recommended Future Business Actions for {selected_date_text}",
        xaxis_title="Business Action",
        yaxis_title="Number of SKUs",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 3 — Revenue Leakage Detection
# ==================================================
with tab3:
    st.header("Revenue Leakage Detection")
    st.info(f"Showing prediction-based output for: {selected_date_text}")

    st.markdown(
        f"**Business Impact:** Identifies predicted revenue leakage risk for **{selected_date_text}**, helping the business focus on margin, stock, and revenue recovery."
    )

    leakage_df = sku_summary.sort_values("Future_Estimated_Revenue_Leakage", ascending=False)

    leakage_count = leakage_df[
        leakage_df["Future_Revenue_Leakage_Flag"] == "Future Leakage Risk"
    ].shape[0]

    leakage_value = leakage_df["Future_Estimated_Revenue_Leakage"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Leakage Risk SKUs on {selected_date_text}", leakage_count)
    c2.metric(f"Estimated Leakage on {selected_date_text}", f"{leakage_value:,.2f}")
    c3.metric(f"Target Margin % on {selected_date_text}", f"{target_margin_pct}%")
    c4.metric(f"Anomaly Risk on {selected_date_text}", "Yes" if future_anomaly_flag == 1 else "No")

    leakage_cols = [
        category_col,
        sku_col,
        "Future_Predicted_Revenue",
        "Future_Predicted_Profit",
        "Future_Predicted_Margin_%",
        "Future_Predicted_Stock",
        "Future_Revenue_Leakage_Flag",
        "Future_Revenue_Leakage_Reason",
        "Future_Estimated_Revenue_Leakage"
    ]

    st.subheader(f"Future Revenue Leakage Risk Table for {selected_date_text}")
    st.dataframe(leakage_df[leakage_cols], use_container_width=True)

    top_leakage = leakage_df.head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_leakage[sku_col].astype(str),
        y=top_leakage["Future_Estimated_Revenue_Leakage"],
        name=f"Estimated Leakage on {selected_date_text}"
    ))

    fig.update_layout(
        title=f"Top 10 SKUs by Future Revenue Leakage for {selected_date_text}",
        xaxis_title="SKU",
        yaxis_title="Estimated Leakage",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 4 — Inventory Optimization Engine
# ==================================================
with tab4:
    st.header("Inventory Optimization Engine")
    st.info(f"Showing prediction-based output for: {selected_date_text}")

    st.markdown(
        f"**Business Impact:** Shows predicted inventory risk for **{selected_date_text}**, helping reduce stockouts, overstock, and holding cost."
    )

    inventory_df = sku_summary.sort_values(
        ["Future_Inventory_Status", "Future_Estimated_Holding_Cost"],
        ascending=[True, False]
    )

    reorder_count = inventory_df[
        inventory_df["Future_Inventory_Status"] == "Future Reorder Required"
    ].shape[0]

    overstock_count = inventory_df[
        inventory_df["Future_Inventory_Status"] == "Future Possible Overstock"
    ].shape[0]

    balanced_count = inventory_df[
        inventory_df["Future_Inventory_Status"] == "Future Balanced"
    ].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Reorder SKUs on {selected_date_text}", reorder_count)
    c2.metric(f"Overstock SKUs on {selected_date_text}", overstock_count)
    c3.metric(f"Balanced SKUs on {selected_date_text}", balanced_count)
    c4.metric(f"Holding Cost on {selected_date_text}", f"{inventory_df['Future_Estimated_Holding_Cost'].sum():,.2f}")

    inventory_cols = [
        category_col,
        sku_col,
        "Future_Predicted_Quantity",
        "Future_Predicted_Stock",
        "Recommended_Safety_Stock",
        "Recommended_Reorder_Point",
        "Future_Inventory_Status",
        "Future_Inventory_Action",
        "Future_Estimated_Holding_Cost"
    ]

    st.subheader(f"Future Inventory Optimisation Table for {selected_date_text}")
    st.dataframe(inventory_df[inventory_cols], use_container_width=True)

    inventory_summary = (
        inventory_df
        .groupby("Future_Inventory_Status")
        .size()
        .reset_index(name="SKU_Count")
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=inventory_summary["Future_Inventory_Status"],
        y=inventory_summary["SKU_Count"],
        name=f"SKU Count on {selected_date_text}"
    ))

    fig.update_layout(
        title=f"Future Inventory Status Summary for {selected_date_text}",
        xaxis_title="Inventory Status",
        yaxis_title="Number of SKUs",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 5 — Executive AI Summary
# ==================================================
with tab5:
    st.header("Executive AI Summary")
    st.info(f"Showing prediction-based output for: {selected_date_text}")

    st.markdown(
        f"**Business Impact:** Provides an executive-level summary of predicted profit, leakage, inventory, and anomaly risk for **{selected_date_text}**."
    )

    future_total_revenue = sku_summary["Future_Predicted_Revenue"].sum()
    future_total_profit = sku_summary["Future_Predicted_Profit"].sum()
    future_avg_margin = sku_summary["Future_Predicted_Margin_%"].mean()
    future_total_leakage = sku_summary["Future_Estimated_Revenue_Leakage"].sum()

    reorder_count = sku_summary[
        sku_summary["Future_Inventory_Status"] == "Future Reorder Required"
    ].shape[0]

    overstock_count = sku_summary[
        sku_summary["Future_Inventory_Status"] == "Future Possible Overstock"
    ].shape[0]

    balanced_count = sku_summary[
        sku_summary["Future_Inventory_Status"] == "Future Balanced"
    ].shape[0]

    best_sku_row = sku_summary.sort_values("Future_Predicted_Profit", ascending=False).head(1)
    leakage_sku_row = sku_summary.sort_values("Future_Estimated_Revenue_Leakage", ascending=False).head(1)

    best_sku = best_sku_row[sku_col].iloc[0] if not best_sku_row.empty else "N/A"
    best_profit = best_sku_row["Future_Predicted_Profit"].iloc[0] if not best_sku_row.empty else 0

    leakage_sku = leakage_sku_row[sku_col].iloc[0] if not leakage_sku_row.empty else "N/A"
    leakage_value = leakage_sku_row["Future_Estimated_Revenue_Leakage"].iloc[0] if not leakage_sku_row.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Revenue on {selected_date_text}", f"{future_total_revenue:,.2f}")
    c2.metric(f"Profit on {selected_date_text}", f"{future_total_profit:,.2f}")
    c3.metric(f"Margin % on {selected_date_text}", f"{future_avg_margin:.2f}%")
    c4.metric(f"Leakage on {selected_date_text}", f"{future_total_leakage:,.2f}")

    st.markdown(f"""
### Executive AI Summary for {selected_date_text}

The selected future date is expected to generate predicted revenue of **{future_total_revenue:,.2f}** and predicted profit of **{future_total_profit:,.2f}**.

The predicted average margin is **{future_avg_margin:.2f}%**.

The estimated future revenue leakage is **{future_total_leakage:,.2f}**.

Future anomaly risk on **{selected_date_text}**: **{"Yes" if future_anomaly_flag == 1 else "No"}**

### Best Commercial Opportunity

The strongest future profit-contributing SKU on **{selected_date_text}** is **{best_sku}**, with predicted profit of **{best_profit:,.2f}**.

Recommended action:
- Prioritise this SKU for promotion.
- Protect stock availability.
- Avoid supply disruption.
- Consider bundle or cross-sell opportunities.

### Revenue Leakage Risk

The SKU with the highest future leakage risk on **{selected_date_text}** is **{leakage_sku}**, with estimated leakage of **{leakage_value:,.2f}**.

Recommended action:
- Review pricing.
- Check cost structure.
- Investigate slow-moving stock.
- Consider promotional clearance.

### Inventory Position on {selected_date_text}

- Future reorder required SKUs: **{reorder_count}**
- Future possible overstock SKUs: **{overstock_count}**
- Future balanced SKUs: **{balanced_count}**

### CEO / CTO Level Recommendation

1. **Grow profitable SKUs**  
   Promote SKUs with high predicted revenue and high predicted profit for **{selected_date_text}**.

2. **Fix future leakage areas**  
   Investigate SKUs with low predicted margin, low predicted revenue, or overstock risk.

3. **Optimise inventory investment**  
   Replenish SKUs with future stockout risk and reduce excess inventory exposure.

4. **Monitor risk periods**  
   If the future anomaly flag is Yes, review demand, stock, margin, and operational KPIs before that period.
""")

    executive_actions = pd.DataFrame({
        "Priority": ["High", "High", "Medium", "Medium", "Low"],
        "Action Area": [
            "Profit Growth",
            "Revenue Leakage",
            "Inventory Optimisation",
            "Anomaly Monitoring",
            "Forecast Governance"
        ],
        "Recommended Action": [
            f"Promote high predicted profit SKUs for {selected_date_text} and protect stock availability",
            f"Investigate SKUs with future leakage risk on {selected_date_text}",
            f"Reorder understocked SKUs and reduce overstock exposure for {selected_date_text}",
            f"Monitor abnormal future KPI behaviour for {selected_date_text}",
            "Review forecast outputs periodically with business users"
        ],
        "Business Benefit": [
            "Revenue and profit uplift",
            "Reduced missed revenue opportunity",
            "Lower holding cost and fewer stockouts",
            "Earlier risk detection",
            "Improved trust in AI-driven planning"
        ]
    })

    st.subheader(f"Top Executive Actions for {selected_date_text}")
    st.dataframe(executive_actions, use_container_width=True)
