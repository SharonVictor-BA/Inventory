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
    page_title="Retail Revenue Leakage & Inventory Optimization App",
    layout="wide"
)

st.title("Retail Revenue Leakage & Inventory Optimization App")

st.markdown("""
This application focuses on three business outcomes:

1. **Revenue Leakage & Inventory Optimization**  
2. **Demand & Forecasting**  
3. **Estimated AI Recommendations**

The backend uses PCA, ARIMA forecasting, and anomaly detection to predict future business behaviour.
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
# Backend Feature Selection
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

pc_df = pd.DataFrame(
    pcs,
    columns=[f"PC{i + 1}" for i in range(n_components)]
)

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
# Backend Anomaly Logic
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
# Future Prediction Selection
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

days_ahead = (pd.to_datetime(selected_future_date).date() - today).days

if days_ahead <= 0:
    selected_forecast_index = 0
else:
    selected_forecast_index = min(days_ahead - 1, len(future_df) - 1)

selected_future_row = future_df.iloc[[selected_forecast_index]]
selected_future_anomaly = future_anomaly_df.iloc[[selected_forecast_index]]
selected_future_values = selected_future_row.iloc[0]

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
    business_df["Business_Quantity"] = pd.to_numeric(
        business_df[quantity_col],
        errors="coerce"
    ).fillna(0)
else:
    business_df["Business_Quantity"] = 1

if revenue_col:
    business_df["Business_Revenue"] = pd.to_numeric(
        business_df[revenue_col],
        errors="coerce"
    ).fillna(0)
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
    business_df["Business_Stock"] = pd.to_numeric(
        business_df[stock_col],
        errors="coerce"
    ).fillna(0)
else:
    business_df["Business_Stock"] = business_df["Business_Quantity"] * 2

if lead_time_col:
    business_df["Business_Lead_Time"] = pd.to_numeric(
        business_df[lead_time_col],
        errors="coerce"
    ).fillna(0)
else:
    business_df["Business_Lead_Time"] = 7

business_df["Business_Profit"] = (
    business_df["Business_Revenue"] - business_df["Business_Cost"]
)

business_df["Business_Margin_%"] = np.where(
    business_df["Business_Revenue"] > 0,
    business_df["Business_Profit"] / business_df["Business_Revenue"] * 100,
    0
)

# --------------------------------------------------
# SKU Summary
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
        Record_Count=("Business_Revenue", "count")
    )
    .reset_index()
)

sku_summary["Revenue_Per_Unit"] = np.where(
    sku_summary["Total_Quantity"] > 0,
    sku_summary["Total_Revenue"] / sku_summary["Total_Quantity"],
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

sku_summary["Predicted_Quantity"] = sku_summary["Total_Quantity"] * future_quantity_factor
sku_summary["Predicted_Revenue"] = sku_summary["Total_Revenue"] * future_revenue_factor
sku_summary["Predicted_Cost"] = sku_summary["Total_Cost"] * future_cost_factor
sku_summary["Predicted_Profit"] = (
    sku_summary["Predicted_Revenue"] - sku_summary["Predicted_Cost"]
)

sku_summary["Predicted_Margin_%"] = np.where(
    sku_summary["Predicted_Revenue"] > 0,
    sku_summary["Predicted_Profit"] / sku_summary["Predicted_Revenue"] * 100,
    0
)

sku_summary["Predicted_Stock"] = sku_summary["Avg_Stock"] * future_stock_factor

# --------------------------------------------------
# Revenue Leakage Logic
# --------------------------------------------------
revenue_threshold = sku_summary["Predicted_Revenue"].median()
stock_threshold = sku_summary["Predicted_Stock"].median()

sku_summary["Revenue_Leakage_Flag"] = np.where(
    (
        (sku_summary["Predicted_Revenue"] < revenue_threshold)
        | (sku_summary["Predicted_Margin_%"] < target_margin_pct)
        | (sku_summary["Predicted_Stock"] > stock_threshold)
    ),
    "Leakage Risk",
    "Healthy"
)

sku_summary["Revenue_Leakage_Reason"] = np.select(
    [
        sku_summary["Predicted_Margin_%"] < target_margin_pct,
        sku_summary["Predicted_Revenue"] < revenue_threshold,
        sku_summary["Predicted_Stock"] > stock_threshold
    ],
    [
        "Predicted margin is below target",
        "Predicted revenue is below median",
        "Predicted stock may be excessive"
    ],
    default="No major leakage signal"
)

sku_summary["Estimated_Revenue_Leakage"] = np.where(
    sku_summary["Revenue_Leakage_Flag"] == "Leakage Risk",
    np.maximum(
        sku_summary["Predicted_Revenue"] * 0.10,
        sku_summary["Predicted_Stock"] * sku_summary["Revenue_Per_Unit"] * 0.05
    ),
    0
)

# --------------------------------------------------
# Inventory Optimization Logic
# --------------------------------------------------
sku_summary["Estimated_Demand"] = (
    sku_summary["Predicted_Quantity"]
    / sku_summary["Record_Count"].replace(0, 1)
)

sku_summary["Recommended_Safety_Stock"] = (
    sku_summary["Estimated_Demand"]
    * (sku_summary["Avg_Lead_Time"] / 7)
    * 0.5
)

sku_summary["Recommended_Reorder_Point"] = (
    sku_summary["Estimated_Demand"]
    * (sku_summary["Avg_Lead_Time"] / 7)
) + sku_summary["Recommended_Safety_Stock"]

sku_summary["Inventory_Status"] = np.select(
    [
        sku_summary["Predicted_Stock"] < sku_summary["Recommended_Reorder_Point"],
        sku_summary["Predicted_Stock"] > sku_summary["Recommended_Reorder_Point"] * 2
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
        "Reduce buying or run promotion"
    ],
    default="Maintain current inventory level"
)

sku_summary["Estimated_Holding_Cost"] = (
    sku_summary["Predicted_Stock"]
    * sku_summary["Revenue_Per_Unit"]
    * holding_cost_pct / 100
)

# --------------------------------------------------
# AI Recommendation Logic
# --------------------------------------------------
sku_summary["Recommended_AI_Action"] = np.select(
    [
        (sku_summary["Revenue_Leakage_Flag"] == "Leakage Risk")
        & (sku_summary["Inventory_Status"] == "Possible Overstock"),

        (sku_summary["Revenue_Leakage_Flag"] == "Leakage Risk")
        & (sku_summary["Inventory_Status"] == "Reorder Required"),

        sku_summary["Revenue_Leakage_Flag"] == "Leakage Risk",

        sku_summary["Inventory_Status"] == "Reorder Required",

        sku_summary["Inventory_Status"] == "Possible Overstock"
    ],
    [
        "Investigate leakage and clear excess stock",
        "Investigate leakage and replenish carefully",
        "Investigate revenue leakage and margin risk",
        "Prioritise replenishment",
        "Reduce buying or clear excess stock"
    ],
    default="Monitor"
)

sku_summary["Priority"] = np.select(
    [
        sku_summary["Recommended_AI_Action"].isin([
            "Investigate leakage and clear excess stock",
            "Investigate leakage and replenish carefully",
            "Investigate revenue leakage and margin risk",
            "Prioritise replenishment"
        ]),
        sku_summary["Recommended_AI_Action"] == "Reduce buying or clear excess stock"
    ],
    [
        "High",
        "Medium"
    ],
    default="Low"
)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1. Revenue Leakage & Inventory Optimization",
    "2. Demand & Forecasting",
    "3. Estimated AI Recommendations"
])

# ==================================================
# TAB 1 — Revenue Leakage & Inventory Optimization
# ==================================================
with tab1:
    st.header("Revenue Leakage & Inventory Optimization")
    st.info("Showing prediction-based output for the selected future period.")

    st.markdown(
        "**Business Impact:** Combines revenue leakage and inventory risk to identify SKUs that may lose revenue, create excess stock, or require replenishment."
    )

    combined_df = sku_summary.sort_values(
        ["Estimated_Revenue_Leakage", "Estimated_Holding_Cost"],
        ascending=[False, False]
    )

    leakage_count = combined_df[combined_df["Revenue_Leakage_Flag"] == "Leakage Risk"].shape[0]
    reorder_count = combined_df[combined_df["Inventory_Status"] == "Reorder Required"].shape[0]
    overstock_count = combined_df[combined_df["Inventory_Status"] == "Possible Overstock"].shape[0]
    total_leakage = combined_df["Estimated_Revenue_Leakage"].sum()
    total_holding_cost = combined_df["Estimated_Holding_Cost"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leakage Risk SKUs", leakage_count)
    c2.metric("Estimated Leakage", f"{total_leakage:,.2f}")
    c3.metric("Reorder SKUs", reorder_count)
    c4.metric("Overstock SKUs", overstock_count)

    c5, c6 = st.columns(2)
    c5.metric("Holding Cost", f"{total_holding_cost:,.2f}")
    c6.metric("Anomaly Risk", "Yes" if future_anomaly_flag == 1 else "No")

    combined_cols = [
        category_col,
        sku_col,
        "Predicted_Revenue",
        "Predicted_Profit",
        "Predicted_Margin_%",
        "Predicted_Stock",
        "Revenue_Leakage_Flag",
        "Revenue_Leakage_Reason",
        "Estimated_Revenue_Leakage",
        "Estimated_Demand",
        "Recommended_Safety_Stock",
        "Recommended_Reorder_Point",
        "Inventory_Status",
        "Inventory_Action",
        "Estimated_Holding_Cost",
        "Recommended_AI_Action",
        "Priority"
    ]

    st.subheader("Combined Revenue Leakage and Inventory Risk Table")
    st.dataframe(combined_df[combined_cols], use_container_width=True)

    top_risk = combined_df.head(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_risk[sku_col].astype(str),
        y=top_risk["Estimated_Revenue_Leakage"],
        name="Estimated Revenue Leakage"
    ))

    fig.add_trace(go.Bar(
        x=top_risk[sku_col].astype(str),
        y=top_risk["Estimated_Holding_Cost"],
        name="Estimated Holding Cost"
    ))

    fig.update_layout(
        title="Top 10 SKUs by Revenue Leakage and Holding Cost Risk",
        xaxis_title="SKU",
        yaxis_title="Estimated Value",
        barmode="group",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    risk_summary = (
        combined_df
        .groupby(["Revenue_Leakage_Flag", "Inventory_Status", "Priority"])
        .size()
        .reset_index(name="SKU_Count")
        .sort_values("SKU_Count", ascending=False)
    )

    st.subheader("Combined Risk Summary")
    st.dataframe(risk_summary, use_container_width=True)

# ==================================================
# TAB 2 — Demand & Forecasting
# ==================================================
with tab2:
    st.header("Demand & Forecasting")
    st.info("Showing prediction-based output for the selected future period.")

    st.markdown(
        "**Business Impact:** Shows predicted demand, revenue, stock, and anomaly signals to support future planning and replenishment decisions."
    )

    total_predicted_quantity = sku_summary["Predicted_Quantity"].sum()
    total_predicted_revenue = sku_summary["Predicted_Revenue"].sum()
    total_predicted_profit = sku_summary["Predicted_Profit"].sum()
    avg_predicted_margin = sku_summary["Predicted_Margin_%"].mean()

    forecast_df = sku_summary.sort_values("Predicted_Quantity", ascending=False)

    forecast_cols = [
        category_col,
        sku_col,
        "Predicted_Quantity",
        "Predicted_Revenue",
        "Predicted_Cost",
        "Predicted_Profit",
        "Predicted_Margin_%",
        "Predicted_Stock",
        "Estimated_Demand",
        "Inventory_Status",
        "Revenue_Leakage_Flag",
        "Priority"
    ]

    st.subheader("Demand and Forecasting Table")
    st.dataframe(forecast_df[forecast_cols], use_container_width=True)

    top_demand = forecast_df.head(10)

    fig_demand = go.Figure()

    fig_demand.add_trace(go.Bar(
        x=top_demand[sku_col].astype(str),
        y=top_demand["Predicted_Quantity"],
        name="Predicted Demand Qty"
    ))

    fig_demand.update_layout(
        title="Top 10 SKUs by Predicted Demand",
        xaxis_title="SKU",
        yaxis_title="Predicted Quantity",
        template="plotly_white"
    )

    st.plotly_chart(fig_demand, use_container_width=True)

    top_revenue = forecast_df.sort_values("Predicted_Revenue", ascending=False).head(10)

    fig_revenue = go.Figure()

    fig_revenue.add_trace(go.Bar(
        x=top_revenue[sku_col].astype(str),
        y=top_revenue["Predicted_Revenue"],
        name="Predicted Revenue"
    ))

    fig_revenue.update_layout(
        title="Top 10 SKUs by Predicted Revenue",
        xaxis_title="SKU",
        yaxis_title="Predicted Revenue",
        template="plotly_white"
    )

    st.plotly_chart(fig_revenue, use_container_width=True)

    anomaly_text = (
        "A future anomaly is predicted. Review demand, revenue, stock, and margin before action."
        if future_anomaly_flag == 1
        else "No major future anomaly is predicted for the selected period."
    )

    st.markdown(f"""
### Forecasting Interpretation

- The forecast predicts total demand of **{total_predicted_quantity:,.0f} units**.
- Expected revenue is **{total_predicted_revenue:,.2f}**.
- Expected profit is **{total_predicted_profit:,.2f}**.
- Average predicted margin is **{avg_predicted_margin:.2f}%**.
- **Anomaly Signal:** {anomaly_text}
""")

# ==================================================
# TAB 3 — Estimated AI Recommendations
# ==================================================
with tab3:
    st.header("Estimated AI Recommendations")
    st.info("Showing prediction-based output for the selected future period.")

    st.markdown(
        "**Business Impact:** Converts forecast, leakage, and inventory results into simple recommended actions for business users."
    )

    total_leakage = sku_summary["Estimated_Revenue_Leakage"].sum()
    total_holding_cost = sku_summary["Estimated_Holding_Cost"].sum()
    leakage_count = sku_summary[sku_summary["Revenue_Leakage_Flag"] == "Leakage Risk"].shape[0]
    reorder_count = sku_summary[sku_summary["Inventory_Status"] == "Reorder Required"].shape[0]
    overstock_count = sku_summary[sku_summary["Inventory_Status"] == "Possible Overstock"].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated Leakage", f"{total_leakage:,.2f}")
    c2.metric("Holding Cost", f"{total_holding_cost:,.2f}")
    c3.metric("High Risk SKUs", leakage_count + reorder_count)
    c4.metric("Anomaly Risk", "Yes" if future_anomaly_flag == 1 else "No")

    recommendation_summary = (
        sku_summary
        .groupby(["Recommended_AI_Action", "Priority"])
        .size()
        .reset_index(name="SKU_Count")
        .sort_values(["Priority", "SKU_Count"], ascending=[True, False])
    )

    st.subheader("AI Recommendation Summary")
    st.dataframe(recommendation_summary, use_container_width=True)

    recommendation_df = sku_summary.sort_values(
        ["Priority", "Estimated_Revenue_Leakage", "Estimated_Holding_Cost"],
        ascending=[True, False, False]
    )

    recommendation_cols = [
        category_col,
        sku_col,
        "Predicted_Revenue",
        "Predicted_Profit",
        "Predicted_Quantity",
        "Predicted_Stock",
        "Revenue_Leakage_Flag",
        "Inventory_Status",
        "Recommended_AI_Action",
        "Priority"
    ]

    st.subheader("SKU-Level AI Recommendations")
    st.dataframe(recommendation_df[recommendation_cols], use_container_width=True)

    highest_leakage_row = sku_summary.sort_values("Estimated_Revenue_Leakage", ascending=False).head(1)
    highest_stock_row = sku_summary.sort_values("Predicted_Stock", ascending=False).head(1)
    highest_demand_row = sku_summary.sort_values("Predicted_Quantity", ascending=False).head(1)

    highest_leakage_sku = highest_leakage_row[sku_col].iloc[0] if not highest_leakage_row.empty else "N/A"
    highest_leakage_value = highest_leakage_row["Estimated_Revenue_Leakage"].iloc[0] if not highest_leakage_row.empty else 0

    highest_stock_sku = highest_stock_row[sku_col].iloc[0] if not highest_stock_row.empty else "N/A"
    highest_stock_value = highest_stock_row["Predicted_Stock"].iloc[0] if not highest_stock_row.empty else 0

    highest_demand_sku = highest_demand_row[sku_col].iloc[0] if not highest_demand_row.empty else "N/A"
    highest_demand_value = highest_demand_row["Predicted_Quantity"].iloc[0] if not highest_demand_row.empty else 0

    st.markdown(f"""
### Executive AI Summary

The system predicts total estimated revenue leakage of **{total_leakage:,.2f}** and inventory holding cost of **{total_holding_cost:,.2f}**.

There are **{leakage_count} SKUs** showing revenue leakage risk, **{reorder_count} SKUs** requiring reorder, and **{overstock_count} SKUs** showing possible overstock risk.

The SKU with the highest estimated leakage risk is **{highest_leakage_sku}**, with possible leakage of **{highest_leakage_value:,.2f}**.

The SKU with the highest predicted stock exposure is **{highest_stock_sku}**, with predicted stock of **{highest_stock_value:,.2f}** units.

The SKU with the highest predicted demand is **{highest_demand_sku}**, with predicted demand of **{highest_demand_value:,.0f}** units.

### Recommended Business Actions

1. Investigate SKUs with high estimated revenue leakage.
2. Review low-margin SKUs and validate pricing or cost assumptions.
3. Replenish SKUs marked as reorder required.
4. Reduce buying or run promotions for overstocked SKUs.
5. Prioritise high-demand SKUs to avoid missed sales.
6. Monitor anomaly risk before executing business decisions.
""")
