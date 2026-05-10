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
    page_title="Smart Retail Executive Forecasting App",
    layout="wide"
)

st.title("Smart Retail Executive Forecasting, Profit & Inventory Intelligence")

st.markdown("""
This application helps business and technical teams convert retail data into **executive-level business decisions**.

It combines:
- PCA-based KPI forecasting
- Future anomaly prediction
- Profit impact simulation
- Revenue leakage detection
- Inventory optimisation
- AI-based executive summary and business recommendations

The objective is to help leadership answer questions such as:

**Which SKUs should we promote?**  
**Where are we losing revenue?**  
**What is the expected profit impact?**  
**Which inventory items need action?**  
**What should the CEO / CTO / business team focus on next?**
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
        "sku" in lower_col or
        "item" in lower_col or
        "product" in lower_col or
        "stockcode" in lower_col
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
else:
    st.sidebar.info("No Date column found. App will use future step numbers.")

if category_col:
    category_options = ["All"] + sorted(filtered_df[category_col].dropna().astype(str).unique())
    selected_category = st.sidebar.selectbox("Select Category", category_options)

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df[category_col].astype(str) == selected_category]
else:
    st.sidebar.info("No Category column found.")

if sku_col:
    sku_options = ["All"] + sorted(filtered_df[sku_col].dropna().astype(str).unique())
    selected_sku = st.sidebar.selectbox("Select SKU / Item / Product", sku_options)

    if selected_sku != "All":
        filtered_df = filtered_df[filtered_df[sku_col].astype(str) == selected_sku]
else:
    st.sidebar.info("No SKU / Item / Product column found.")

if filtered_df.empty:
    st.error("No records available for the selected filters.")
    st.stop()

st.sidebar.success(f"Filtered records: {len(filtered_df)}")

# --------------------------------------------------
# Numeric Columns and Business Column Selection
# --------------------------------------------------
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 3:
    st.error("Dataset must contain at least 3 numeric columns for PCA analysis.")
    st.stop()

default_features = [
    col for col in [
        "sales_qty",
        "sales_revenue",
        "lead_time_days",
        "delivery_reliability",
        "obsolescence_risk",
        "stock_on_hand",
        "unit_cost",
        "unit_price",
        "profit"
    ] if col in numeric_cols
]

if len(default_features) < 3:
    default_features = numeric_cols[:5]

features = st.sidebar.multiselect(
    "Select KPI Features for PCA Forecasting",
    numeric_cols,
    default=default_features
)

if len(features) < 3:
    st.error("Please select at least 3 numeric KPI features.")
    st.stop()

st.sidebar.header("Business Assumption Settings")

quantity_col = st.sidebar.selectbox(
    "Select Quantity Column",
    options=["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index("sales_qty") if "sales_qty" in numeric_cols else 0
)

revenue_col = st.sidebar.selectbox(
    "Select Revenue Column",
    options=["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index("sales_revenue") if "sales_revenue" in numeric_cols else 0
)

cost_col = st.sidebar.selectbox(
    "Select Cost Column",
    options=["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index("unit_cost") if "unit_cost" in numeric_cols else 0
)

price_col = st.sidebar.selectbox(
    "Select Unit Price Column",
    options=["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index("unit_price") if "unit_price" in numeric_cols else 0
)

stock_col = st.sidebar.selectbox(
    "Select Stock / Inventory Column",
    options=["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index("stock_on_hand") if "stock_on_hand" in numeric_cols else 0
)

lead_time_col = st.sidebar.selectbox(
    "Select Lead Time Column",
    options=["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index("lead_time_days") if "lead_time_days" in numeric_cols else 0
)

service_level_col = st.sidebar.selectbox(
    "Select Delivery / Reliability Column",
    options=["None"] + numeric_cols,
    index=(["None"] + numeric_cols).index("delivery_reliability") if "delivery_reliability" in numeric_cols else 0
)

future_steps = st.sidebar.slider(
    "Future Prediction Periods",
    min_value=5,
    max_value=30,
    value=10
)

threshold_percentile = st.sidebar.slider(
    "Anomaly Threshold Percentile",
    min_value=90,
    max_value=99,
    value=95
)

expected_uplift_pct = st.sidebar.slider(
    "Expected Sales Uplift % for Profit Simulator",
    min_value=1,
    max_value=100,
    value=10
)

target_margin_pct = st.sidebar.slider(
    "Target Margin % for Business Rules",
    min_value=1,
    max_value=90,
    value=25
)

holding_cost_pct = st.sidebar.slider(
    "Inventory Holding Cost % Estimate",
    min_value=1,
    max_value=50,
    value=10
)

# --------------------------------------------------
# Prepare Data
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

# --------------------------------------------------
# Forecast PCs using ARIMA
# --------------------------------------------------
pc_forecasts = {}
pc_lower = {}
pc_upper = {}

for col in pc_df.columns:
    try:
        model = ARIMA(pc_df[col], order=(2, 1, 2))
        model_fit = model.fit()

        forecast_result = model_fit.get_forecast(steps=future_steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

        pc_forecasts[col] = forecast_mean.values
        pc_lower[col] = conf_int.iloc[:, 0].values
        pc_upper[col] = conf_int.iloc[:, 1].values

    except Exception:
        last_value = pc_df[col].iloc[-1]
        std_value = pc_df[col].std()

        pc_forecasts[col] = np.repeat(last_value, future_steps)
        pc_lower[col] = np.repeat(last_value - 1.96 * std_value, future_steps)
        pc_upper[col] = np.repeat(last_value + 1.96 * std_value, future_steps)

pc_forecast_df = pd.DataFrame(pc_forecasts)
pc_lower_df = pd.DataFrame(pc_lower)
pc_upper_df = pd.DataFrame(pc_upper)

future_scaled = pca.inverse_transform(pc_forecast_df)
future_values = scaler.inverse_transform(future_scaled)
future_df = pd.DataFrame(future_values, columns=features)

future_lower_scaled = pca.inverse_transform(pc_lower_df)
future_upper_scaled = pca.inverse_transform(pc_upper_df)

future_lower_values = scaler.inverse_transform(future_lower_scaled)
future_upper_values = scaler.inverse_transform(future_upper_scaled)

future_lower_df = pd.DataFrame(
    future_lower_values,
    columns=[f"{col}_Lower_95" for col in features]
)

future_upper_df = pd.DataFrame(
    future_upper_values,
    columns=[f"{col}_Upper_95" for col in features]
)

# --------------------------------------------------
# Future Dates / Steps
# --------------------------------------------------
if date_col:
    last_date = filtered_df[date_col].max()
    inferred_freq = pd.infer_freq(filtered_df[date_col].dropna())

    if inferred_freq is None:
        inferred_freq = "W"

    future_dates = pd.date_range(
        start=last_date,
        periods=future_steps + 1,
        freq=inferred_freq
    )[1:]

    future_df.insert(0, "Date", future_dates)
    future_lower_df.insert(0, "Date", future_dates)
    future_upper_df.insert(0, "Date", future_dates)

    current_values = filtered_df[[date_col] + features].tail(future_steps).reset_index(drop=True)
    current_values.rename(columns={date_col: "Date"}, inplace=True)

else:
    future_df.insert(0, "Step", np.arange(1, future_steps + 1))
    future_lower_df.insert(0, "Step", np.arange(1, future_steps + 1))
    future_upper_df.insert(0, "Step", np.arange(1, future_steps + 1))

    current_values = X.tail(future_steps).reset_index(drop=True)
    current_values.insert(0, "Step", np.arange(1, future_steps + 1))

# --------------------------------------------------
# Monitoring Metrics
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
    (spe > spe_threshold) |
    (t2 > t2_threshold) |
    (g2 > g2_threshold)
)

results = pd.DataFrame({
    "SPE": spe,
    "T2": t2,
    "G2": g2,
    "SPE_Threshold": spe_threshold,
    "T2_Threshold": t2_threshold,
    "G2_Threshold": g2_threshold,
    "Anomaly": anomaly.astype(int)
})

# --------------------------------------------------
# Future Anomaly Prediction
# --------------------------------------------------
future_pcs = pc_forecast_df.values
future_scaled_reconstructed = pca.inverse_transform(future_pcs)
future_residual = future_scaled - future_scaled_reconstructed

future_spe = np.sum(future_residual ** 2, axis=1)
future_t2 = np.sum((future_pcs ** 2) / eigen_vals, axis=1)

future_spe_norm = (future_spe - np.min(spe)) / (np.max(spe) - np.min(spe) + 1e-9)
future_t2_norm = (future_t2 - np.min(t2)) / (np.max(t2) - np.min(t2) + 1e-9)
future_g2 = 0.5 * future_spe_norm + 0.5 * future_t2_norm

future_anomaly = (
    (future_spe > spe_threshold) |
    (future_t2 > t2_threshold) |
    (future_g2 > g2_threshold)
)

future_anomaly_df = pd.DataFrame({
    "SPE_Forecast": future_spe,
    "T2_Forecast": future_t2,
    "G2_Forecast": future_g2,
    "Future_Anomaly": future_anomaly.astype(int)
})

if "Date" in future_df.columns:
    future_anomaly_df.insert(0, "Date", future_df["Date"])
else:
    future_anomaly_df.insert(0, "Step", future_df["Step"])

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def safe_col(col):
    return col if col != "None" else None


quantity_col = safe_col(quantity_col)
revenue_col = safe_col(revenue_col)
cost_col = safe_col(cost_col)
price_col = safe_col(price_col)
stock_col = safe_col(stock_col)
lead_time_col = safe_col(lead_time_col)
service_level_col = safe_col(service_level_col)


def create_business_base_table(data):
    business_df = data.copy()

    if sku_col is None:
        business_df["SKU"] = "Unknown SKU"
        local_sku_col = "SKU"
    else:
        local_sku_col = sku_col

    if category_col is None:
        business_df["Category"] = "Unknown Category"
        local_category_col = "Category"
    else:
        local_category_col = category_col

    if quantity_col:
        business_df["Business_Quantity"] = pd.to_numeric(business_df[quantity_col], errors="coerce").fillna(0)
    else:
        business_df["Business_Quantity"] = 1

    if revenue_col:
        business_df["Business_Revenue"] = pd.to_numeric(business_df[revenue_col], errors="coerce").fillna(0)
    elif price_col:
        business_df["Business_Revenue"] = (
            pd.to_numeric(business_df[price_col], errors="coerce").fillna(0) *
            business_df["Business_Quantity"]
        )
    else:
        business_df["Business_Revenue"] = 0

    if cost_col:
        business_df["Business_Cost"] = (
            pd.to_numeric(business_df[cost_col], errors="coerce").fillna(0) *
            business_df["Business_Quantity"]
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

    if service_level_col:
        business_df["Business_Service_Level"] = pd.to_numeric(business_df[service_level_col], errors="coerce").fillna(0)
    else:
        business_df["Business_Service_Level"] = 95

    business_df["Business_Profit"] = business_df["Business_Revenue"] - business_df["Business_Cost"]

    business_df["Business_Margin_%"] = np.where(
        business_df["Business_Revenue"] > 0,
        business_df["Business_Profit"] / business_df["Business_Revenue"] * 100,
        0
    )

    return business_df, local_sku_col, local_category_col


business_df, local_sku_col, local_category_col = create_business_base_table(filtered_df)

sku_business_summary = (
    business_df
    .groupby([local_category_col, local_sku_col], dropna=False)
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

sku_business_summary["Expected_Uplift_Revenue"] = (
    sku_business_summary["Total_Revenue"] * expected_uplift_pct / 100
)

sku_business_summary["Expected_Uplift_Profit"] = (
    sku_business_summary["Total_Profit"] * expected_uplift_pct / 100
)

sku_business_summary["Revenue_Per_Unit"] = np.where(
    sku_business_summary["Total_Quantity"] > 0,
    sku_business_summary["Total_Revenue"] / sku_business_summary["Total_Quantity"],
    0
)

sku_business_summary["Profit_Per_Unit"] = np.where(
    sku_business_summary["Total_Quantity"] > 0,
    sku_business_summary["Total_Profit"] / sku_business_summary["Total_Quantity"],
    0
)

# --------------------------------------------------
# Revenue Leakage Logic
# --------------------------------------------------
revenue_threshold = sku_business_summary["Total_Revenue"].median()
margin_threshold = target_margin_pct
stock_threshold = sku_business_summary["Avg_Stock"].median()

sku_business_summary["Revenue_Leakage_Flag"] = np.where(
    (
        (sku_business_summary["Total_Revenue"] < revenue_threshold) |
        (sku_business_summary["Avg_Margin_Percent"] < margin_threshold) |
        (sku_business_summary["Avg_Stock"] > stock_threshold)
    ),
    "Leakage Risk",
    "Healthy"
)

sku_business_summary["Revenue_Leakage_Reason"] = np.select(
    [
        sku_business_summary["Avg_Margin_Percent"] < margin_threshold,
        sku_business_summary["Total_Revenue"] < revenue_threshold,
        sku_business_summary["Avg_Stock"] > stock_threshold
    ],
    [
        "Low margin compared to target",
        "Low revenue contribution",
        "Possible excess stock / slow movement"
    ],
    default="No major leakage signal"
)

sku_business_summary["Estimated_Revenue_Leakage"] = np.where(
    sku_business_summary["Revenue_Leakage_Flag"] == "Leakage Risk",
    np.maximum(
        sku_business_summary["Total_Revenue"] * 0.10,
        sku_business_summary["Avg_Stock"] * sku_business_summary["Revenue_Per_Unit"] * 0.05
    ),
    0
)

# --------------------------------------------------
# Inventory Optimisation Logic
# --------------------------------------------------
sku_business_summary["Estimated_Demand_Next_Period"] = (
    sku_business_summary["Total_Quantity"] / sku_business_summary["Record_Count"].replace(0, 1)
)

sku_business_summary["Recommended_Safety_Stock"] = (
    sku_business_summary["Estimated_Demand_Next_Period"] *
    (sku_business_summary["Avg_Lead_Time"] / 7) *
    0.5
)

sku_business_summary["Recommended_Reorder_Point"] = (
    sku_business_summary["Estimated_Demand_Next_Period"] *
    (sku_business_summary["Avg_Lead_Time"] / 7)
) + sku_business_summary["Recommended_Safety_Stock"]

sku_business_summary["Inventory_Status"] = np.select(
    [
        sku_business_summary["Avg_Stock"] < sku_business_summary["Recommended_Reorder_Point"],
        sku_business_summary["Avg_Stock"] > sku_business_summary["Recommended_Reorder_Point"] * 2
    ],
    [
        "Reorder Required",
        "Possible Overstock"
    ],
    default="Balanced"
)

sku_business_summary["Inventory_Action"] = np.select(
    [
        sku_business_summary["Inventory_Status"] == "Reorder Required",
        sku_business_summary["Inventory_Status"] == "Possible Overstock"
    ],
    [
        "Increase replenishment / avoid stockout",
        "Reduce purchase quantity / consider promotion"
    ],
    default="Maintain current inventory level"
)

sku_business_summary["Estimated_Holding_Cost"] = (
    sku_business_summary["Avg_Stock"] *
    sku_business_summary["Revenue_Per_Unit"] *
    holding_cost_pct / 100
)

# --------------------------------------------------
# Business Action Recommendation Logic
# --------------------------------------------------
sku_business_summary["Business_Action"] = np.select(
    [
        (
            (sku_business_summary["Total_Revenue"] >= revenue_threshold) &
            (sku_business_summary["Avg_Margin_Percent"] >= margin_threshold)
        ),
        (
            (sku_business_summary["Total_Revenue"] >= revenue_threshold) &
            (sku_business_summary["Avg_Margin_Percent"] < margin_threshold)
        ),
        (
            (sku_business_summary["Revenue_Leakage_Flag"] == "Leakage Risk") &
            (sku_business_summary["Inventory_Status"] == "Possible Overstock")
        ),
        (
            sku_business_summary["Inventory_Status"] == "Reorder Required"
        )
    ],
    [
        "Promote / Prioritise SKU",
        "Review pricing or cost",
        "Clear excess stock with promotion",
        "Replenish inventory"
    ],
    default="Monitor"
)

sku_business_summary["Action_Priority"] = np.select(
    [
        sku_business_summary["Business_Action"].isin(["Promote / Prioritise SKU", "Replenish inventory"]),
        sku_business_summary["Business_Action"].isin(["Review pricing or cost", "Clear excess stock with promotion"])
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "1. Prediction",
    "2. PCA",
    "3. Monitoring",
    "4. Anomaly Detection",
    "5. Profit Impact Simulator",
    "6. Business Action Recommendations",
    "7. Revenue Leakage Detection",
    "8. Inventory Optimization Engine",
    "9. Executive AI Summary"
])

# ==================================================
# TAB 1 — Prediction
# ==================================================
with tab1:
    st.header("Future KPI Prediction")

    st.markdown("""
This tab forecasts future KPI values beyond the current dataset.  
The model forecasts PCA components first and then reconstructs future KPI values.
""")

    selected_kpi = st.selectbox("Select KPI to compare", features)

    x_current = current_values["Date"] if "Date" in current_values.columns else current_values["Step"]
    x_future = future_df["Date"] if "Date" in future_df.columns else future_df["Step"]

    lower_col = f"{selected_kpi}_Lower_95"
    upper_col = f"{selected_kpi}_Upper_95"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_current,
        y=current_values[selected_kpi],
        mode="lines+markers",
        name=f"Current {selected_kpi}"
    ))

    fig.add_trace(go.Scatter(
        x=x_future,
        y=future_df[selected_kpi],
        mode="lines+markers",
        name=f"Future {selected_kpi}"
    ))

    fig.add_trace(go.Scatter(
        x=x_future,
        y=future_upper_df[upper_col],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper 95%"
    ))

    fig.add_trace(go.Scatter(
        x=x_future,
        y=future_lower_df[lower_col],
        mode="lines",
        fill="tonexty",
        line=dict(width=0),
        name="95% Confidence Band"
    ))

    fig.update_layout(
        title=f"Current vs Future Forecast with Confidence Band: {selected_kpi}",
        xaxis_title="Date" if "Date" in future_df.columns else "Future Step",
        yaxis_title=selected_kpi,
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    selected_future_mean = future_df[selected_kpi].mean()
    selected_current_mean = current_values[selected_kpi].mean()

    if selected_future_mean > selected_current_mean:
        trend_text = "Future forecast is expected to increase compared to recent historical values."
    elif selected_future_mean < selected_current_mean:
        trend_text = "Future forecast is expected to decrease compared to recent historical values."
    else:
        trend_text = "Future forecast is expected to remain broadly stable."

    st.markdown(f"""
### Forecast Insight

- Average current value: **{selected_current_mean:.2f}**
- Average future value: **{selected_future_mean:.2f}**
- Interpretation: **{trend_text}**

**Business Meaning:**  
Use this forecast to plan stock, pricing, promotion, replenishment, and operational capacity.
""")

    st.subheader("Future KPI Forecast Table")

    forecast_display_df = pd.concat(
        [
            future_df,
            future_lower_df.drop(columns=["Date"], errors="ignore").drop(columns=["Step"], errors="ignore"),
            future_upper_df.drop(columns=["Date"], errors="ignore").drop(columns=["Step"], errors="ignore")
        ],
        axis=1
    )

    st.dataframe(forecast_display_df, use_container_width=True)

# ==================================================
# TAB 2 — PCA
# ==================================================
with tab2:
    st.header("Principal Component Analysis")

    variance_df = pd.DataFrame({
        "Principal Component": pc_names,
        "Variance Explained (%)": np.round(pca.explained_variance_ratio_ * 100, 2)
    })

    st.subheader("PCA Variance Explained")
    st.dataframe(variance_df, use_container_width=True)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=variance_df["Principal Component"],
        y=variance_df["Variance Explained (%)"]
    ))

    fig_bar.update_layout(
        title="Variance Explained by PCA Components",
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    if n_components >= 2:
        st.subheader("PCA Scatter Plot")

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=pc_df["PC1"],
            y=pc_df["PC2"],
            mode="markers",
            marker=dict(size=7),
            name="Observations"
        ))

        fig_scatter.update_layout(
            title="PCA Projection: PC1 vs PC2",
            xaxis_title="PC1",
            yaxis_title="PC2",
            template="plotly_white"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("""
### Business Meaning of PCA

- **PC1** usually represents the strongest demand / revenue pattern.
- **PC2** usually captures supply, lead-time, or operational variation.
- **PC3** usually captures secondary operational instability.

This helps reduce multiple KPIs into fewer business intelligence signals.
""")

# ==================================================
# TAB 3 — Monitoring
# ==================================================
with tab3:
    st.header("PCA-Based Monitoring")

    fig_monitor = go.Figure()

    fig_monitor.add_trace(go.Scatter(
        y=results["SPE"],
        mode="lines",
        name="SPE"
    ))

    fig_monitor.add_trace(go.Scatter(
        y=results["T2"],
        mode="lines",
        name="T²"
    ))

    fig_monitor.add_trace(go.Scatter(
        y=results["G2"],
        mode="lines",
        name="G₂"
    ))

    fig_monitor.update_layout(
        title="PCA Monitoring Metrics: SPE, T² and G₂",
        xaxis_title="Index",
        yaxis_title="Monitoring Score",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig_monitor, use_container_width=True)

    st.markdown(f"""
### Monitoring Insight

- Maximum SPE score: **{results["SPE"].max():.2f}**
- Maximum T² score: **{results["T2"].max():.2f}**
- Maximum G₂ score: **{results["G2"].max():.2f}**

**Business Meaning:**  
Rising monitoring scores indicate demand, supply, revenue, or operational behaviour moving away from normal patterns.
""")

# ==================================================
# TAB 4 — Anomaly Detection
# ==================================================
with tab4:
    st.header("Anomaly Detection and AI Recommendations")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(results))
    c2.metric("Anomalies Detected", int(results["Anomaly"].sum()))
    c3.metric("Anomaly Rate (%)", round(results["Anomaly"].mean() * 100, 2))
    c4.metric("Threshold Percentile", threshold_percentile)

    st.subheader("Anomaly Detection Results")
    st.dataframe(results, use_container_width=True)

    fig_alarm = go.Figure()

    fig_alarm.add_trace(go.Scatter(
        y=results["Anomaly"],
        mode="lines+markers",
        name="Alarm",
        line_shape="hv"
    ))

    fig_alarm.update_layout(
        title="Alarm Trigger: 0 = Normal, 1 = Anomaly",
        xaxis_title="Index",
        yaxis_title="Alarm",
        template="plotly_white"
    )

    st.plotly_chart(fig_alarm, use_container_width=True)

    anomaly_rate = results["Anomaly"].mean() * 100

    if anomaly_rate > 20:
        st.warning("""
High anomaly rate detected.

Recommended actions:
- Review demand spikes and sales volatility
- Check supplier lead-time issues
- Investigate delivery reliability
- Increase monitoring frequency
- Consider safety stock adjustment
""")
    elif anomaly_rate > 5:
        st.info("""
Moderate anomaly rate detected.

Recommended actions:
- Monitor high-risk KPIs closely
- Review demand and supply-related drivers
- Validate replenishment planning assumptions
- Track recurring abnormal periods
""")
    else:
        st.success("""
Low anomaly rate detected.

Recommended actions:
- Continue routine monitoring
- Maintain current inventory strategy
- Use forecasts for proactive replenishment planning
""")

# ==================================================
# TAB 5 — Profit Impact Simulator
# ==================================================
with tab5:
    st.header("Profit Impact Simulator")

    st.markdown("""
This tab estimates the potential profit impact if selected SKUs receive additional sales uplift.

It helps answer:

**If we promote or recommend this SKU, what is the possible revenue and profit upside?**
""")

    top_profit_df = sku_business_summary.sort_values("Total_Profit", ascending=False).copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{sku_business_summary['Total_Revenue'].sum():,.2f}")
    c2.metric("Total Profit", f"{sku_business_summary['Total_Profit'].sum():,.2f}")
    c3.metric("Avg Margin %", f"{sku_business_summary['Avg_Margin_Percent'].mean():.2f}%")
    c4.metric("Sales Uplift Assumption", f"{expected_uplift_pct}%")

    st.subheader("SKU-Level Profit Impact Simulation")

    profit_cols = [
        local_category_col,
        local_sku_col,
        "Total_Quantity",
        "Total_Revenue",
        "Total_Cost",
        "Total_Profit",
        "Avg_Margin_Percent",
        "Expected_Uplift_Revenue",
        "Expected_Uplift_Profit"
    ]

    st.dataframe(
        top_profit_df[profit_cols].head(50),
        use_container_width=True
    )

    fig_profit = go.Figure()

    top_10_profit = top_profit_df.head(10)

    fig_profit.add_trace(go.Bar(
        x=top_10_profit[local_sku_col].astype(str),
        y=top_10_profit["Total_Profit"],
        name="Current Profit"
    ))

    fig_profit.add_trace(go.Bar(
        x=top_10_profit[local_sku_col].astype(str),
        y=top_10_profit["Expected_Uplift_Profit"],
        name="Expected Additional Profit"
    ))

    fig_profit.update_layout(
        title="Top 10 SKUs: Current Profit vs Expected Additional Profit",
        xaxis_title="SKU",
        yaxis_title="Profit",
        barmode="group",
        template="plotly_white"
    )

    st.plotly_chart(fig_profit, use_container_width=True)

    st.markdown("""
### Executive Interpretation

- SKUs with high current profit and high expected uplift should be prioritised for promotion.
- SKUs with poor margin should be reviewed before increasing sales volume.
- This simulator helps estimate the commercial benefit before launching a campaign.
""")

# ==================================================
# TAB 6 — Business Action Recommendation Table
# ==================================================
with tab6:
    st.header("Business Action Recommendation Table")

    st.markdown("""
This tab converts analytical results into simple business actions.

It helps business users understand:

**Which SKUs should be promoted, repriced, replenished, cleared, or monitored?**
""")

    action_df = sku_business_summary.sort_values(
        ["Action_Priority", "Total_Profit"],
        ascending=[True, False]
    ).copy()

    action_cols = [
        local_category_col,
        local_sku_col,
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

    fig_action = go.Figure()

    fig_action.add_trace(go.Bar(
        x=action_summary["Business_Action"],
        y=action_summary["SKU_Count"],
        name="SKU Count"
    ))

    fig_action.update_layout(
        title="Recommended Business Actions by SKU Count",
        xaxis_title="Business Action",
        yaxis_title="Number of SKUs",
        template="plotly_white"
    )

    st.plotly_chart(fig_action, use_container_width=True)

    st.markdown("""
### Business Meaning

- **Promote / Prioritise SKU:** Strong revenue and margin.
- **Review pricing or cost:** Revenue is high, but margin is weak.
- **Clear excess stock with promotion:** Inventory may be blocking working capital.
- **Replenish inventory:** Demand exists but stock may be insufficient.
- **Monitor:** No urgent action required.
""")

# ==================================================
# TAB 7 — Revenue Leakage Detection
# ==================================================
with tab7:
    st.header("Revenue Leakage Detection")

    st.markdown("""
This tab identifies SKUs where the business may be losing revenue due to:

- Low revenue contribution
- Low margin
- Excess stock
- Poor conversion or slow movement

It helps answer:

**Where is revenue or profit potentially leaking?**
""")

    leakage_df = sku_business_summary.sort_values(
        "Estimated_Revenue_Leakage",
        ascending=False
    ).copy()

    leakage_count = leakage_df[leakage_df["Revenue_Leakage_Flag"] == "Leakage Risk"].shape[0]
    leakage_value = leakage_df["Estimated_Revenue_Leakage"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Leakage Risk SKUs", leakage_count)
    c2.metric("Estimated Leakage Value", f"{leakage_value:,.2f}")
    c3.metric("Target Margin %", f"{target_margin_pct}%")

    leakage_cols = [
        local_category_col,
        local_sku_col,
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

    fig_leakage = go.Figure()

    fig_leakage.add_trace(go.Bar(
        x=top_leakage[local_sku_col].astype(str),
        y=top_leakage["Estimated_Revenue_Leakage"],
        name="Estimated Leakage"
    ))

    fig_leakage.update_layout(
        title="Top 10 SKUs by Estimated Revenue Leakage",
        xaxis_title="SKU",
        yaxis_title="Estimated Revenue Leakage",
        template="plotly_white"
    )

    st.plotly_chart(fig_leakage, use_container_width=True)

    st.markdown("""
### Business Meaning

Revenue leakage does not always mean actual accounting loss.  
It indicates **missed commercial opportunity** or **risk areas** where the business should investigate.

Typical actions:
- Review price and margin
- Reduce excess stock
- Promote slow-moving items
- Improve replenishment planning
- Investigate demand drop
""")

# ==================================================
# TAB 8 — Inventory Optimization Engine
# ==================================================
with tab8:
    st.header("Inventory Optimization Engine")

    st.markdown("""
This tab estimates whether SKUs are understocked, overstocked, or balanced.

It helps answer:

**Which SKUs need replenishment?**  
**Which SKUs may be overstocked?**  
**Where is working capital locked in inventory?**
""")

    inventory_df = sku_business_summary.sort_values(
        ["Inventory_Status", "Estimated_Holding_Cost"],
        ascending=[True, False]
    ).copy()

    reorder_count = inventory_df[inventory_df["Inventory_Status"] == "Reorder Required"].shape[0]
    overstock_count = inventory_df[inventory_df["Inventory_Status"] == "Possible Overstock"].shape[0]
    balanced_count = inventory_df[inventory_df["Inventory_Status"] == "Balanced"].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reorder Required SKUs", reorder_count)
    c2.metric("Possible Overstock SKUs", overstock_count)
    c3.metric("Balanced SKUs", balanced_count)
    c4.metric("Estimated Holding Cost", f"{inventory_df['Estimated_Holding_Cost'].sum():,.2f}")

    inventory_cols = [
        local_category_col,
        local_sku_col,
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

    fig_inventory = go.Figure()

    fig_inventory.add_trace(go.Bar(
        x=inventory_summary["Inventory_Status"],
        y=inventory_summary["SKU_Count"],
        name="SKU Count"
    ))

    fig_inventory.update_layout(
        title="Inventory Status Summary",
        xaxis_title="Inventory Status",
        yaxis_title="Number of SKUs",
        template="plotly_white"
    )

    st.plotly_chart(fig_inventory, use_container_width=True)

    st.markdown("""
### Business Meaning

- **Reorder Required:** Stock may not be enough to support expected demand.
- **Possible Overstock:** Stock may be higher than demand requirement.
- **Balanced:** Current stock appears reasonable against demand and lead-time assumptions.

This is useful for reducing stockout risk and avoiding unnecessary inventory holding cost.
""")

# ==================================================
# TAB 9 — Executive AI Summary
# ==================================================
with tab9:
    st.header("Executive AI Summary")

    st.markdown("""
This tab converts the analysis into an executive-level business summary for CEO, CTO, Product, Finance, Supply Chain, and Operations teams.
""")

    total_revenue = sku_business_summary["Total_Revenue"].sum()
    total_profit = sku_business_summary["Total_Profit"].sum()
    avg_margin = sku_business_summary["Avg_Margin_Percent"].mean()
    total_leakage = sku_business_summary["Estimated_Revenue_Leakage"].sum()
    anomaly_count = int(results["Anomaly"].sum())
    anomaly_rate = results["Anomaly"].mean() * 100
    future_anomaly_count = int(future_anomaly_df["Future_Anomaly"].sum())

    best_sku_row = sku_business_summary.sort_values("Total_Profit", ascending=False).head(1)
    highest_leakage_row = sku_business_summary.sort_values("Estimated_Revenue_Leakage", ascending=False).head(1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{total_revenue:,.2f}")
    c2.metric("Total Profit", f"{total_profit:,.2f}")
    c3.metric("Average Margin %", f"{avg_margin:.2f}%")
    c4.metric("Estimated Leakage", f"{total_leakage:,.2f}")

    st.subheader("Executive Summary")

    if not best_sku_row.empty:
        best_sku = best_sku_row[local_sku_col].iloc[0]
        best_profit = best_sku_row["Total_Profit"].iloc[0]
    else:
        best_sku = "N/A"
        best_profit = 0

    if not highest_leakage_row.empty:
        leakage_sku = highest_leakage_row[local_sku_col].iloc[0]
        leakage_value = highest_leakage_row["Estimated_Revenue_Leakage"].iloc[0]
    else:
        leakage_sku = "N/A"
        leakage_value = 0

    st.markdown(f"""
### 1. Overall Business Position

The analysed dataset generated total revenue of **{total_revenue:,.2f}** and total profit of **{total_profit:,.2f}**.  
The average margin across SKUs is **{avg_margin:.2f}%**.

### 2. Best Commercial Opportunity

The strongest profit-contributing SKU is **{best_sku}**, contributing approximately **{best_profit:,.2f}** in profit.

Recommended action:
- Prioritise this SKU for campaign planning
- Maintain stock availability
- Avoid supply disruption
- Consider cross-sell or bundle opportunities

### 3. Revenue Leakage Risk

The SKU with the highest estimated revenue leakage is **{leakage_sku}**, with estimated leakage of **{leakage_value:,.2f}**.

Recommended action:
- Review pricing
- Check cost structure
- Investigate slow-moving stock
- Consider promotion or stock clearance

### 4. Inventory Position

- SKUs requiring reorder: **{reorder_count}**
- SKUs with possible overstock: **{overstock_count}**
- Balanced SKUs: **{balanced_count}**

Recommended action:
- Replenish understocked high-demand SKUs
- Reduce future buying for overstocked SKUs
- Use promotions to release working capital from slow-moving inventory

### 5. Risk and Anomaly Position

Historical anomaly count: **{anomaly_count}**  
Historical anomaly rate: **{anomaly_rate:.2f}%**  
Future anomaly periods predicted: **{future_anomaly_count}**

Recommended action:
- Monitor abnormal demand, revenue, or supply behaviour
- Investigate periods with high SPE, T², or G₂ scores
- Use the future anomaly signal as an early warning indicator

### 6. CEO / CTO Level Recommendation

The business should focus on three priorities:

1. **Grow profitable SKUs**  
   Promote SKUs with strong profit and margin.

2. **Fix leakage areas**  
   Investigate low-margin, low-revenue, and overstocked SKUs.

3. **Optimise inventory investment**  
   Reduce excess stock while protecting high-demand SKUs from stockout risk.
""")

    st.subheader("Top Recommended Executive Actions")

    executive_actions = pd.DataFrame({
        "Priority": [
            "High",
            "High",
            "Medium",
            "Medium",
            "Low"
        ],
        "Action Area": [
            "Profit Growth",
            "Revenue Leakage",
            "Inventory Optimisation",
            "Anomaly Monitoring",
            "Forecast Governance"
        ],
        "Recommended Action": [
            "Promote high-profit SKUs and protect stock availability",
            "Investigate SKUs with low margin, low revenue, or excess stock",
            "Reorder understocked SKUs and reduce overstock exposure",
            "Monitor abnormal KPI behaviour using SPE, T² and G₂",
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

    st.dataframe(executive_actions, use_container_width=True)
