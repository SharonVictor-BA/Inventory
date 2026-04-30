import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="PCA Retail Forecasting & Monitoring",
    layout="wide"
)

st.title("Smart Retail PCA Forecasting, Monitoring & Anomaly Detection")

st.markdown("""
This application uses **PCA-based intelligence** to analyse retail inventory behaviour.

It supports:
- Future KPI prediction
- PCA interpretation
- Monitoring using SPE, T² and G₂
- Anomaly detection
- Category and SKU-level business impact analysis
""")

# --------------------------------------------------
# Upload Data
# --------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# Detect Category, SKU and Date columns
# --------------------------------------------------
category_col = None
sku_col = None
date_col = None

for col in df.columns:
    lower_col = col.lower()

    if category_col is None and "category" in lower_col:
        category_col = col

    if sku_col is None and ("sku" in lower_col or "item" in lower_col or "product" in lower_col):
        sku_col = col

    if date_col is None and "date" in lower_col:
        date_col = col

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col)

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("Filters")

filtered_df = df.copy()

if category_col:
    category_options = ["All"] + sorted(filtered_df[category_col].dropna().astype(str).unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", category_options)

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df[category_col].astype(str) == selected_category]
else:
    st.sidebar.info("No Category column found.")

if sku_col:
    sku_options = ["All"] + sorted(filtered_df[sku_col].dropna().astype(str).unique().tolist())
    selected_sku = st.sidebar.selectbox("Select SKU_ID", sku_options)

    if selected_sku != "All":
        filtered_df = filtered_df[filtered_df[sku_col].astype(str) == selected_sku]
else:
    st.sidebar.info("No SKU_ID / Item / Product column found.")

if filtered_df.empty:
    st.error("No records available for the selected filters.")
    st.stop()

st.sidebar.success(f"Filtered records: {len(filtered_df)}")

# --------------------------------------------------
# Numeric KPI Selection
# --------------------------------------------------
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()

default_features = [
    col for col in [
        "sales_qty",
        "sales_revenue",
        "lead_time_days",
        "delivery_reliability",
        "obsolescence_risk"
    ] if col in numeric_cols
]

if len(default_features) < 3:
    default_features = numeric_cols[:5]

features = st.sidebar.multiselect(
    "Select KPI Features",
    numeric_cols,
    default=default_features
)

if len(features) < 3:
    st.error("Please select at least 3 numeric KPI features.")
    st.stop()

impact_kpi = st.sidebar.selectbox(
    "Select KPI for Top / Bottom SKU Impact",
    features
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

# --------------------------------------------------
# Prepare Data
# --------------------------------------------------
X = filtered_df[features].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

if len(X) < 10:
    st.error("Not enough records after filtering. Please select a broader category or SKU.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = min(3, len(features))
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(X_scaled)

pc_names = [f"PC{i+1}" for i in range(n_components)]
pc_df = pd.DataFrame(pcs, columns=pc_names)

# --------------------------------------------------
# Forecast PCs using ARIMA + Confidence Intervals
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
# Future Dates
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

if date_col:
    future_anomaly_df.insert(0, "Date", future_dates)
else:
    future_anomaly_df.insert(0, "Step", np.arange(1, future_steps + 1))

# --------------------------------------------------
# Top / Bottom SKU Business Impact
# --------------------------------------------------
def get_sku_impact_table(data, sku_col, category_col, kpi):
    if sku_col is None:
        return None, None

    group_cols = [sku_col]
    if category_col and category_col in data.columns:
        group_cols.insert(0, category_col)

    sku_summary = (
        data.groupby(group_cols)[kpi]
        .agg(["mean", "sum", "min", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    top_3 = sku_summary.head(3)
    bottom_3 = sku_summary.tail(3).sort_values("mean", ascending=True)

    return top_3, bottom_3

top_skus, bottom_skus = get_sku_impact_table(
    filtered_df,
    sku_col,
    category_col,
    impact_kpi
)

# --------------------------------------------------
# PCA Explanation
# --------------------------------------------------
pc_explanation = {
    "PC1": {
        "name": "Demand Intelligence",
        "meaning": "Captures sales intensity, demand movement, revenue behaviour, and profitability signals.",
        "business_value": "Helps identify demand spikes, demand drops, and revenue-impacting changes."
    },
    "PC2": {
        "name": "Supply Intelligence",
        "meaning": "Captures lead-time behaviour, supplier reliability, and supply-side volatility.",
        "business_value": "Helps detect supplier delay, logistics risk, and replenishment instability."
    },
    "PC3": {
        "name": "Operational Intelligence",
        "meaning": "Captures operational stability, execution consistency, and recurring process deviations.",
        "business_value": "Helps identify operational inefficiencies and service-level degradation."
    }
}

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Prediction",
    "2. PCA",
    "3. Monitoring",
    "4. Anomaly Detection"
])

# ==================================================
# TAB 1 — PREDICTION
# ==================================================
with tab1:
    st.header("Future KPI Prediction")

    st.markdown("""
This tab forecasts future KPI values beyond the current dataset.
The model forecasts PCA components first and then reconstructs future KPI values.
""")

    selected_kpi = st.selectbox("Select KPI to compare", features)

    x_current = current_values["Date"] if date_col else current_values["Step"]
    x_future = future_df["Date"] if date_col else future_df["Step"]

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
        xaxis_title="Date" if date_col else "Step",
        yaxis_title=selected_kpi,
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

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

    st.subheader("Future Anomaly Prediction")
    st.dataframe(future_anomaly_df, use_container_width=True)

    future_anomaly_count = int(future_anomaly_df["Future_Anomaly"].sum())

    if future_anomaly_count > 0:
        st.warning(f"{future_anomaly_count} future periods are predicted as potential anomalies.")
    else:
        st.success("No future anomaly is predicted in the selected forecast window.")

    st.subheader("Top / Bottom SKU Impact")

    if sku_col and top_skus is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"### Top 3 SKU_ID by High `{impact_kpi}`")
            st.dataframe(top_skus, use_container_width=True)

        with c2:
            st.markdown(f"### Bottom 3 SKU_ID by Low `{impact_kpi}`")
            st.dataframe(bottom_skus, use_container_width=True)

        st.info(
            "These tables show which SKUs have the highest and lowest business impact for the selected KPI "
            "within the selected category/SKU filter."
        )
    else:
        st.info("SKU_ID column not found, so Top / Bottom SKU impact table cannot be created.")

# ==================================================
# TAB 2 — PCA
# ==================================================
with tab2:
    st.header("Principal Component Analysis")

    st.markdown("""
PCA reduces multiple retail KPIs into fewer meaningful components.
This supports interpretation across demand, supply, and operational behaviour.
""")

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

    st.subheader("PCA Component Meaning")

    for pc in pc_names:
        info = pc_explanation.get(pc)
        if info:
            st.markdown(f"""
### {pc} — {info['name']}
- **Meaning:** {info['meaning']}
- **Business Value:** {info['business_value']}
""")

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

# ==================================================
# TAB 3 — MONITORING
# ==================================================
with tab3:
    st.header("PCA-Based Monitoring")

    st.markdown("""
This tab tracks PCA-based monitoring signals.
The threshold lines separate normal behaviour from potential abnormal behaviour.
""")

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

    st.markdown("""
### Graph Explanation
- **SPE** detects sudden deviations from normal behaviour
- **T²** detects structural shifts in PCA space
- **G₂** captures persistent or subtle anomalies
""")

# ==================================================
# TAB 4 — ANOMALY DETECTION
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

    st.subheader("AI Recommendations")

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
