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
    page_title="Smart Retail Executive Intelligence App",
    layout="wide"
)

st.title("Smart Retail Executive Intelligence App")

st.markdown("""
This application converts retail transaction and inventory data into executive-level business insights.

The app focuses on:
- Profit impact simulation
- Business action recommendations
- Revenue leakage detection
- Inventory optimisation
- Executive AI summary

Backend intelligence such as PCA forecasting, anomaly detection and risk classification runs behind the scenes.
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

if date_col:
    min_date = filtered_df[date_col].min().date()
    max_date = filtered_df[date_col].max().date()

    selected_date_range = st.sidebar.date_input(
        "Select Historical Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        filtered_df = filtered_df[
            (filtered_df[date_col].dt.date >= start_date) &
            (filtered_df[date_col].dt.date <= end_date)
        ]

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
# Numeric Column Setup
# --------------------------------------------------
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 3:
    st.error("Dataset must contain at least 3 numeric columns.")
    st.stop()

st.sidebar.header("Column Mapping")

def get_default_column(possible_names):
    for name in possible_names:
        for col in numeric_cols:
            if name.lower() in col.lower():
                return col
    return "None"

quantity_col = st.sidebar.selectbox(
    "Quantity Column",
    ["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index(get_default_column(["quantity", "qty", "sales_qty"]))
)

revenue_col = st.sidebar.selectbox(
    "Revenue Column",
    ["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index(get_default_column(["revenue", "sales_revenue", "amount"]))
)

cost_col = st.sidebar.selectbox(
    "Cost Column",
    ["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index(get_default_column(["cost", "unit_cost"]))
)

price_col = st.sidebar.selectbox(
    "Unit Price Column",
    ["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index(get_default_column(["price", "unit_price"]))
)

stock_col = st.sidebar.selectbox(
    "Stock / Inventory Column",
    ["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index(get_default_column(["stock", "inventory", "soh"]))
)

lead_time_col = st.sidebar.selectbox(
    "Lead Time Column",
    ["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index(get_default_column(["lead_time", "lead time"]))
)

service_col = st.sidebar.selectbox(
    "Delivery Reliability / Service Column",
    ["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index(get_default_column(["delivery", "reliability", "service"]))
)

# --------------------------------------------------
# Business Assumptions
# --------------------------------------------------
st.sidebar.header("Business Assumptions")

future_steps = st.sidebar.slider("Future Prediction Periods", 5, 30, 10)

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
# Feature Selection for Backend PCA
# --------------------------------------------------
default_features = numeric_cols[:5]

features = st.sidebar.multiselect(
    "Backend KPI Features for PCA / Forecasting",
    numeric_cols,
    default=default_features
)

if len(features) < 3:
    st.error("Please select at least 3 numeric KPI features.")
    st.stop()

# --------------------------------------------------
# Backend PCA, Forecasting and Anomaly Logic
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

# Forecast PCA components
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

# Monitoring metrics
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
    (spe > spe_threshold) |
    (t2 > t2_threshold) |
    (g2 > g2_threshold)
)

results = pd.DataFrame({
    "SPE": spe,
    "T2": t2,
    "G2": g2,
    "Anomaly": anomaly.astype(int)
})

# Future anomaly prediction
future_pcs = pc_forecast_df.values
future_scaled_reconstructed = pca.inverse_transform(future_pcs)
future_residual = future_scaled - future_scaled_reconstructed

future_spe = np.sum(future_residual ** 2, axis=1)
future_t2 = np.sum((future_pcs ** 2) / eigen_vals, axis=1)

future_g2 = 0.5 * (
    (future_spe - np.min(spe)) / (np.max(spe) - np.min(spe) + 1e-9)
) + 0.5 * (
    (future_t2 - np.min(t2)) / (np.max(t2) - np.min(t2) + 1e-9)
)

future_anomaly = (
    (future_spe > spe_threshold) |
    (future_t2 > t2_threshold) |
    (future_g2 > g2_threshold)
)

future_anomaly_df = pd.DataFrame({
    "Future_SPE": future_spe,
    "Future_T2": future_t2,
    "Future_G2": future_g2,
    "Future_Anomaly": future_anomaly.astype(int)
})

# --------------------------------------------------
# Business Data Preparation
# --------------------------------------------------
def clean_col(col):
    return None if col == "None" else col

quantity_col = clean_col(quantity_col)
revenue_col = clean_col(revenue_col)
cost_col = clean_col(cost_col)
price_col = clean_col(price_col)
stock_col = clean_col(stock_col)
lead_time_col = clean_col(lead_time_col)
service_col = clean_col(service_col)

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
# SKU Business Summary
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

sku_summary["Expected_Uplift_Revenue"] = (
    sku_summary["Total_Revenue"] * expected_uplift_pct / 100
)

sku_summary["Expected_Uplift_Profit"] = (
    sku_summary["Total_Profit"] * expected_uplift_pct / 100
)

# --------------------------------------------------
# Revenue Leakage Logic
# --------------------------------------------------
revenue_threshold = sku_summary["Total_Revenue"].median()
stock_threshold = sku_summary["Avg_Stock"].median()

sku_summary["Revenue_Leakage_Flag"] = np.where(
    (
        (sku_summary["Total_Revenue"] < revenue_threshold) |
        (sku_summary["Avg_Margin_Percent"] < target_margin_pct) |
        (sku_summary["Avg_Stock"] > stock_threshold)
    ),
    "Leakage Risk",
    "Healthy"
)

sku_summary["Revenue_Leakage_Reason"] = np.select(
    [
        sku_summary["Avg_Margin_Percent"] < target_margin_pct,
        sku_summary["Total_Revenue"] < revenue_threshold,
        sku_summary["Avg_Stock"] > stock_threshold
    ],
    [
        "Low margin compared to target",
        "Low revenue contribution",
        "Possible excess stock or slow movement"
    ],
    default="No major leakage signal"
)

sku_summary["Estimated_Revenue_Leakage"] = np.where(
    sku_summary["Revenue_Leakage_Flag"] == "Leakage Risk",
    np.maximum(
        sku_summary["Total_Revenue"] * 0.10,
        sku_summary["Avg_Stock"] * sku_summary["Revenue_Per_Unit"] * 0.05
    ),
    0
)

# --------------------------------------------------
# Inventory Optimisation Logic
# --------------------------------------------------
sku_summary["Estimated_Demand_Next_Period"] = (
    sku_summary["Total_Quantity"] / sku_summary["Record_Count"].replace(0, 1)
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

sku_summary["Inventory_Status"] = np.select(
    [
        sku_summary["Avg_Stock"] < sku_summary["Recommended_Reorder_Point"],
        sku_summary["Avg_Stock"] > sku_summary["Recommended_Reorder_Point"] * 2
    ],
    [
        "Reorder Required",
        "Possible Overstock"
    ],
    default="Balanced"
)

sku_summary["Inventory_Action"] = np.select(
    [
        sku_summary["Inventory_Status"] == "Reorder Required",
        sku_summary["Inventory_Status"] == "Possible Overstock"
    ],
    [
        "Increase replenishment to avoid stockout",
        "Reduce purchase quantity or run promotion"
    ],
    default="Maintain current inventory level"
)

sku_summary["Estimated_Holding_Cost"] = (
    sku_summary["Avg_Stock"]
    * sku_summary["Revenue_Per_Unit"]
    * holding_cost_pct / 100
)

# --------------------------------------------------
# Business Action Logic
# --------------------------------------------------
sku_summary["Business_Action"] = np.select(
    [
        (
            (sku_summary["Total_Revenue"] >= revenue_threshold) &
            (sku_summary["Avg_Margin_Percent"] >= target_margin_pct)
        ),
        (
            (sku_summary["Total_Revenue"] >= revenue_threshold) &
            (sku_summary["Avg_Margin_Percent"] < target_margin_pct)
        ),
        (
            (sku_summary["Revenue_Leakage_Flag"] == "Leakage Risk") &
            (sku_summary["Inventory_Status"] == "Possible Overstock")
        ),
        sku_summary["Inventory_Status"] == "Reorder Required"
    ],
    [
        "Promote / Prioritise SKU",
        "Review pricing or cost",
        "Clear excess stock with promotion",
        "Replenish inventory"
    ],
    default="Monitor"
)

sku_summary["Action_Priority"] = np.select(
    [
        sku_summary["Business_Action"].isin(["Promote / Prioritise SKU", "Replenish inventory"]),
        sku_summary["Business_Action"].isin(["Review pricing or cost", "Clear excess stock with promotion"])
    ],
    [
        "High",
        "Medium"
    ],
    default="Low"
)

# --------------------------------------------------
# Visible Tabs Only
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

    st.markdown("""
This tab estimates the revenue and profit impact if selected SKUs receive additional sales uplift.
""")

    total_revenue = sku_summary["Total_Revenue"].sum()
    total_profit = sku_summary["Total_Profit"].sum()
    avg_margin = sku_summary["Avg_Margin_Percent"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{total_revenue:,.2f}")
    c2.metric("Total Profit", f"{total_profit:,.2f}")
    c3.metric("Average Margin %", f"{avg_margin:.2f}%")
    c4.metric("Sales Uplift Assumption", f"{expected_uplift_pct}%")

    profit_df = sku_summary.sort_values("Total_Profit", ascending=False)

    profit_cols = [
        category_col,
        sku_col,
        "Total_Quantity",
        "Total_Revenue",
        "Total_Cost",
        "Total_Profit",
        "Avg_Margin_Percent",
        "Expected_Uplift_Revenue",
        "Expected_Uplift_Profit"
    ]

    st.subheader("SKU-Level Profit Impact Table")
    st.dataframe(profit_df[profit_cols], use_container_width=True)

    top_10 = profit_df.head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_10[sku_col].astype(str),
        y=top_10["Total_Profit"],
        name="Current Profit"
    ))

    fig.add_trace(go.Bar(
        x=top_10[sku_col].astype(str),
        y=top_10["Expected_Uplift_Profit"],
        name="Expected Additional Profit"
    ))

    fig.update_layout(
        title="Top 10 SKUs: Current Profit vs Expected Additional Profit",
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
    st.header("Business Action Recommendation Table")

    st.markdown("""
This tab converts analytical signals into clear business actions.
""")

    action_df = sku_summary.sort_values(
        ["Action_Priority", "Total_Profit"],
        ascending=[True, False]
    )

    action_cols = [
        category_col,
        sku_col,
        "Total_Revenue",
        "Total_Profit",
        "Avg_Margin_Percent",
        "Avg_Stock",
        "Inventory_Status",
        "Revenue_Leakage_Flag",
        "Business_Action",
        "Action_Priority"
    ]

    st.dataframe(action_df[action_cols], use_container_width=True)

    action_summary = (
        action_df
        .groupby(["Business_Action", "Action_Priority"])
        .size()
        .reset_index(name="SKU_Count")
        .sort_values("SKU_Count", ascending=False)
    )

    st.subheader("Action Summary")
    st.dataframe(action_summary, use_container_width=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=action_summary["Business_Action"],
        y=action_summary["SKU_Count"],
        name="SKU Count"
    ))

    fig.update_layout(
        title="Recommended Business Actions by SKU Count",
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

    st.markdown("""
This tab identifies SKUs where the business may be losing revenue or profit opportunity.
""")

    leakage_df = sku_summary.sort_values("Estimated_Revenue_Leakage", ascending=False)

    leakage_count = leakage_df[leakage_df["Revenue_Leakage_Flag"] == "Leakage Risk"].shape[0]
    leakage_value = leakage_df["Estimated_Revenue_Leakage"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Leakage Risk SKUs", leakage_count)
    c2.metric("Estimated Leakage Value", f"{leakage_value:,.2f}")
    c3.metric("Target Margin %", f"{target_margin_pct}%")

    leakage_cols = [
        category_col,
        sku_col,
        "Total_Revenue",
        "Total_Profit",
        "Avg_Margin_Percent",
        "Avg_Stock",
        "Revenue_Leakage_Flag",
        "Revenue_Leakage_Reason",
        "Estimated_Revenue_Leakage"
    ]

    st.subheader("Revenue Leakage Risk Table")
    st.dataframe(leakage_df[leakage_cols], use_container_width=True)

    top_leakage = leakage_df.head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_leakage[sku_col].astype(str),
        y=top_leakage["Estimated_Revenue_Leakage"],
        name="Estimated Leakage"
    ))

    fig.update_layout(
        title="Top 10 SKUs by Estimated Revenue Leakage",
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

    st.markdown("""
This tab identifies understock, overstock and balanced inventory positions.
""")

    inventory_df = sku_summary.sort_values(
        ["Inventory_Status", "Estimated_Holding_Cost"],
        ascending=[True, False]
    )

    reorder_count = inventory_df[inventory_df["Inventory_Status"] == "Reorder Required"].shape[0]
    overstock_count = inventory_df[inventory_df["Inventory_Status"] == "Possible Overstock"].shape[0]
    balanced_count = inventory_df[inventory_df["Inventory_Status"] == "Balanced"].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reorder Required SKUs", reorder_count)
    c2.metric("Possible Overstock SKUs", overstock_count)
    c3.metric("Balanced SKUs", balanced_count)
    c4.metric("Estimated Holding Cost", f"{inventory_df['Estimated_Holding_Cost'].sum():,.2f}")

    inventory_cols = [
        category_col,
        sku_col,
        "Estimated_Demand_Next_Period",
        "Avg_Stock",
        "Avg_Lead_Time",
        "Recommended_Safety_Stock",
        "Recommended_Reorder_Point",
        "Inventory_Status",
        "Inventory_Action",
        "Estimated_Holding_Cost"
    ]

    st.subheader("Inventory Optimisation Table")
    st.dataframe(inventory_df[inventory_cols], use_container_width=True)

    inventory_summary = (
        inventory_df
        .groupby("Inventory_Status")
        .size()
        .reset_index(name="SKU_Count")
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=inventory_summary["Inventory_Status"],
        y=inventory_summary["SKU_Count"],
        name="SKU Count"
    ))

    fig.update_layout(
        title="Inventory Status Summary",
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

    total_revenue = sku_summary["Total_Revenue"].sum()
    total_profit = sku_summary["Total_Profit"].sum()
    avg_margin = sku_summary["Avg_Margin_Percent"].mean()
    total_leakage = sku_summary["Estimated_Revenue_Leakage"].sum()

    anomaly_count = int(results["Anomaly"].sum())
    anomaly_rate = results["Anomaly"].mean() * 100
    future_anomaly_count = int(future_anomaly_df["Future_Anomaly"].sum())

    best_sku_row = sku_summary.sort_values("Total_Profit", ascending=False).head(1)
    leakage_sku_row = sku_summary.sort_values("Estimated_Revenue_Leakage", ascending=False).head(1)

    best_sku = best_sku_row[sku_col].iloc[0] if not best_sku_row.empty else "N/A"
    best_profit = best_sku_row["Total_Profit"].iloc[0] if not best_sku_row.empty else 0

    leakage_sku = leakage_sku_row[sku_col].iloc[0] if not leakage_sku_row.empty else "N/A"
    leakage_value = leakage_sku_row["Estimated_Revenue_Leakage"].iloc[0] if not leakage_sku_row.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{total_revenue:,.2f}")
    c2.metric("Total Profit", f"{total_profit:,.2f}")
    c3.metric("Average Margin %", f"{avg_margin:.2f}%")
    c4.metric("Estimated Leakage", f"{total_leakage:,.2f}")

    st.markdown(f"""
### Overall Business Position

The analysed dataset generated total revenue of **{total_revenue:,.2f}** and total profit of **{total_profit:,.2f}**.  
The average margin across SKUs is **{avg_margin:.2f}%**.

### Best Commercial Opportunity

The strongest profit-contributing SKU is **{best_sku}**, contributing approximately **{best_profit:,.2f}** in profit.

Recommended actions:
- Prioritise this SKU for promotion
- Maintain stock availability
- Avoid supply disruption
- Consider bundle or cross-sell opportunities

### Revenue Leakage Risk

The SKU with the highest estimated leakage is **{leakage_sku}**, with estimated leakage of **{leakage_value:,.2f}**.

Recommended actions:
- Review pricing
- Check cost structure
- Investigate slow-moving stock
- Consider promotional clearance

### Inventory Position

- Reorder required SKUs: **{reorder_count}**
- Possible overstock SKUs: **{overstock_count}**
- Balanced SKUs: **{balanced_count}**

### Risk and Anomaly Position

- Historical anomaly count: **{anomaly_count}**
- Historical anomaly rate: **{anomaly_rate:.2f}%**
- Future anomaly periods predicted: **{future_anomaly_count}**

### CEO / CTO Level Recommendation

The business should focus on three priorities:

1. **Grow profitable SKUs**  
   Promote high-profit and high-margin SKUs.

2. **Fix leakage areas**  
   Investigate low-margin, low-revenue and overstocked SKUs.

3. **Optimise inventory investment**  
   Reduce excess stock while protecting high-demand SKUs from stockout risk.
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
            "Promote high-profit SKUs and protect stock availability",
            "Investigate SKUs with low margin, low revenue or excess stock",
            "Reorder understocked SKUs and reduce overstock exposure",
            "Monitor abnormal KPI behaviour using backend anomaly signals",
            "Review model outputs periodically with business users"
        ],
        "Business Benefit": [
            "Revenue and profit uplift",
            "Reduced missed revenue opportunity",
            "Lower holding cost and fewer stockouts",
            "Earlier risk detection",
            "Improved trust in AI-driven planning"
        ]
    })

    st.subheader("Top Executive Actions")
    st.dataframe(executive_actions, use_container_width=True)
