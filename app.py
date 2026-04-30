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
    page_title="Smart Retail PCA Forecasting & Monitoring",
    layout="wide"
)

st.title("Smart Retail PCA Forecasting, Monitoring & Anomaly Detection")

st.markdown("""
This app uses **PCA-based intelligence** to analyse retail inventory behaviour.

It supports:
- Future KPI prediction with confidence bands
- PCA interpretation
- Monitoring using SPE, T² and G₂
- Future anomaly prediction
- Category and SKU-level business impact analysis
- AI-based recommendations
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

    if sku_col is None and ("sku" in lower_col or "item" in lower_col or "product" in lower_col):
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

# Historical Date Filter
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

# Category Filter
if category_col:
    category_options = ["All"] + sorted(filtered_df[category_col].dropna().astype(str).unique())
    selected_category = st.sidebar.selectbox("Select Category", category_options)

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df[category_col].astype(str) == selected_category]
else:
    st.sidebar.info("No Category column found.")

# SKU Filter
if sku_col:
    sku_options = ["All"] + sorted(filtered_df[sku_col].dropna().astype(str).unique())
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
# KPI Selection
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

# Convert forecasted PCs back to original KPI space
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
# Top / Bottom SKU Impact
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

    # Future Date or Step Selection
    if "Date" in future_df.columns:
        st.subheader("Future Date Selection")

        future_df["Date_only"] = pd.to_datetime(future_df["Date"]).dt.date
        future_anomaly_df["Date_only"] = pd.to_datetime(future_anomaly_df["Date"]).dt.date

        min_future_date = future_df["Date_only"].min()
        max_future_date = future_df["Date_only"].max()

        selected_future_date = st.date_input(
            "Select Future Date for Prediction Insight",
            value=min_future_date,
            min_value=min_future_date,
            max_value=max_future_date
        )

        selected_future_row = future_df[
            future_df["Date_only"] == selected_future_date
        ].drop(columns=["Date_only"], errors="ignore")

        selected_future_anomaly = future_anomaly_df[
            future_anomaly_df["Date_only"] == selected_future_date
        ].drop(columns=["Date_only"], errors="ignore")

    else:
        st.subheader("Future Step Selection")

        selected_future_step = st.selectbox(
            "Select Future Step for Prediction Insight",
            future_df["Step"].tolist()
        )

        selected_future_row = future_df[
            future_df["Step"] == selected_future_step
        ]

        selected_future_anomaly = future_anomaly_df[
            future_anomaly_df["Step"] == selected_future_step
        ]

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

    # Graph explanation under Prediction graph
    selected_future_mean = future_df[selected_kpi].mean()
    selected_current_mean = current_values[selected_kpi].mean()

    future_upper_mean = future_upper_df[upper_col].mean()
    future_lower_mean = future_lower_df[lower_col].mean()
    confidence_width = future_upper_mean - future_lower_mean

    if selected_future_mean > selected_current_mean:
        trend_text = "Future forecast is expected to increase compared to recent historical values."
    elif selected_future_mean < selected_current_mean:
        trend_text = "Future forecast is expected to decrease compared to recent historical values."
    else:
        trend_text = "Future forecast is expected to remain broadly stable."

    if confidence_width > abs(selected_future_mean) * 0.5:
        uncertainty_text = "The confidence band is wide, indicating higher uncertainty in the prediction."
    else:
        uncertainty_text = "The confidence band is relatively narrow, indicating more stable prediction confidence."

    st.markdown(f"""
### Graph Insight

This graph compares recent historical values with future predicted values for **{selected_kpi}**.

**What it shows**
- Blue line = recent historical KPI values
- Red line = future predicted KPI values
- Shaded area = 95% confidence band

**AI Interpretation**
- Average current value: **{selected_current_mean:.2f}**
- Average future value: **{selected_future_mean:.2f}**
- Forecast uncertainty range: **{future_lower_mean:.2f} to {future_upper_mean:.2f}**
- {trend_text}
- {uncertainty_text}

**Business Meaning**
- Use the forecast line for expected planning.
- Use the confidence band for risk planning.
- Wider band means higher uncertainty and greater need for buffer planning.
""")

    st.subheader("Selected Future Prediction Summary")

    if not selected_future_row.empty:
        st.dataframe(selected_future_row, use_container_width=True)

        if not selected_future_anomaly.empty:
            future_alarm = int(selected_future_anomaly["Future_Anomaly"].iloc[0])

            if future_alarm == 1:
                st.warning("Potential anomaly predicted for the selected future period.")
            else:
                st.success("No anomaly predicted for the selected future period.")
    else:
        st.info("No prediction available for the selected future period.")

    st.subheader("Future KPI Forecast Table")

    forecast_display_df = pd.concat(
        [
            future_df.drop(columns=["Date_only"], errors="ignore"),
            future_lower_df.drop(columns=["Date"], errors="ignore").drop(columns=["Step"], errors="ignore"),
            future_upper_df.drop(columns=["Date"], errors="ignore").drop(columns=["Step"], errors="ignore")
        ],
        axis=1
    )

    st.dataframe(forecast_display_df, use_container_width=True)

    st.subheader("Future Anomaly Prediction")

    st.dataframe(
        future_anomaly_df.drop(columns=["Date_only"], errors="ignore"),
        use_container_width=True
    )

    future_anomaly_count = int(future_anomaly_df["Future_Anomaly"].sum())

    if future_anomaly_count > 0:
        st.warning(f"{future_anomaly_count} future periods are predicted as potential anomalies.")
    else:
        st.success("No future anomaly is predicted in the selected forecast window.")

    st.subheader("Future Anomaly Risk Graph")

    x_future_anom = future_anomaly_df["Date"] if "Date" in future_anomaly_df.columns else future_anomaly_df["Step"]

    fig_anom = go.Figure()

    fig_anom.add_trace(go.Scatter(
        x=x_future_anom,
        y=future_anomaly_df["SPE_Forecast"],
        mode="lines+markers",
        name="Future SPE"
    ))

    fig_anom.add_trace(go.Scatter(
        x=x_future_anom,
        y=future_anomaly_df["T2_Forecast"],
        mode="lines+markers",
        name="Future T²"
    ))

    fig_anom.add_trace(go.Scatter(
        x=x_future_anom,
        y=future_anomaly_df["G2_Forecast"],
        mode="lines+markers",
        name="Future G₂"
    ))

    fig_anom.add_trace(go.Scatter(
        x=x_future_anom,
        y=future_anomaly_df["Future_Anomaly"],
        mode="lines+markers",
        name="Future Alarm",
        yaxis="y2"
    ))

    fig_anom.update_layout(
        title="Future Anomaly Prediction using Forecasted PCA Scores",
        xaxis_title="Date" if "Date" in future_anomaly_df.columns else "Future Step",
        yaxis=dict(title="Anomaly Scores"),
        yaxis2=dict(
            title="Alarm",
            overlaying="y",
            side="right",
            range=[-0.1, 1.1]
        ),
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig_anom, use_container_width=True)

    future_anomaly_rate = future_anomaly_df["Future_Anomaly"].mean() * 100

    st.markdown(f"""
### Graph Insight

This graph predicts whether future periods are likely to behave normally or abnormally.

**What it shows**
- Future SPE, T², and G₂ values estimate future anomaly risk.
- Future Alarm = 1 means potential abnormal behaviour.
- Future Alarm = 0 means expected normal behaviour.

**AI Interpretation**
- Future anomaly periods detected: **{int(future_anomaly_df["Future_Anomaly"].sum())}**
- Future anomaly rate: **{future_anomaly_rate:.2f}%**

**Business Meaning**
- If future anomaly risk is high, review demand, supply, and operational KPIs early.
- Use this graph as an early-warning signal before actual disruption happens.
""")

    st.subheader("AI Recommendations for Selected Future Period")

    if not selected_future_row.empty:
        row = selected_future_row.iloc[0]
        recommendations = []

        if "sales_qty" in row:
            if row["sales_qty"] > future_df["sales_qty"].mean():
                recommendations.append("Demand is expected to be above average. Increase stock availability for high-demand SKUs.")
            else:
                recommendations.append("Demand is expected to remain below or near average. Avoid unnecessary overstock.")

        if "sales_revenue" in row:
            if row["sales_revenue"] > future_df["sales_revenue"].mean():
                recommendations.append("Revenue is expected to be strong. Prioritize high-value SKUs and avoid stockouts.")
            else:
                recommendations.append("Revenue is not expected to exceed average. Review promotion or pricing opportunities.")

        if "lead_time_days" in row:
            if row["lead_time_days"] > future_df["lead_time_days"].mean():
                recommendations.append("Lead time is expected to increase. Review supplier commitments and plan replenishment earlier.")
            else:
                recommendations.append("Lead time is expected to remain stable. Continue normal replenishment planning.")

        if "delivery_reliability" in row:
            if row["delivery_reliability"] < future_df["delivery_reliability"].mean():
                recommendations.append("Delivery reliability is expected to weaken. Monitor logistics risk and prepare backup options.")
            else:
                recommendations.append("Delivery reliability looks stable. Maintain current logistics approach.")

        if "obsolescence_risk" in row:
            if row["obsolescence_risk"] > future_df["obsolescence_risk"].mean():
                recommendations.append("Obsolescence risk is expected to rise. Reduce excess stock or consider promotions.")
            else:
                recommendations.append("Obsolescence risk is expected to remain controlled.")

        if not selected_future_anomaly.empty and int(selected_future_anomaly["Future_Anomaly"].iloc[0]) == 1:
            recommendations.append("Future anomaly risk detected. Review demand, supply, and operational KPIs before this period.")

        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success("Future KPI behaviour looks stable. Continue routine monitoring.")

    st.subheader("Top / Bottom SKU Impact")

    if sku_col and top_skus is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"### Top 3 SKU_ID by High `{impact_kpi}`")
            st.dataframe(top_skus, use_container_width=True)

        with c2:
            st.markdown(f"### Bottom 3 SKU_ID by Low `{impact_kpi}`")
            st.dataframe(bottom_skus, use_container_width=True)

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

    total_variance = variance_df["Variance Explained (%)"].sum()
    top_pc = variance_df.sort_values("Variance Explained (%)", ascending=False).iloc[0]

    st.markdown(f"""
### Graph Insight

This graph shows how much information each principal component captures from the original KPI data.

**What it shows**
- PC1, PC2, and PC3 summarize the original KPI features.
- Higher variance means the component explains more data behaviour.
- Total variance retained: **{total_variance:.2f}%**

**AI Interpretation**
- The most influential component is **{top_pc["Principal Component"]}**, explaining **{top_pc["Variance Explained (%)"]:.2f}%** of the data pattern.

**Business Meaning**
- PC1 usually represents the strongest business driver.
- PC2 and PC3 capture secondary supply and operational behaviour.
- Together, the PCs simplify complex retail data into interpretable intelligence layers.
""")

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

        st.markdown("""
### Graph Insight

This scatter plot shows observations projected into PCA space.

**What it shows**
- Each point represents one record after dimensionality reduction.
- Points close together have similar KPI behaviour.
- Points far away may indicate unusual patterns or different business behaviour.

**AI Interpretation**
- Spread in the plot indicates variation across demand, supply, or operational conditions.
- Isolated points may require further investigation as potential anomalies.

**Business Meaning**
- Helps understand whether retail behaviour is stable or highly varied.
- Supports early identification of unusual SKU or category patterns.
""")

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

    spe_max = results["SPE"].max()
    t2_max = results["T2"].max()
    g2_max = results["G2"].max()

    st.markdown(f"""
### Graph Insight

This graph tracks PCA monitoring scores over time.

**What it shows**
- **SPE** detects sudden deviations from normal behaviour.
- **T²** detects structural changes in PCA space.
- **G₂** captures persistent or subtle abnormal behaviour.

**AI Interpretation**
- Maximum SPE score: **{spe_max:.2f}**
- Maximum T² score: **{t2_max:.2f}**
- Maximum G₂ score: **{g2_max:.2f}**

**Business Meaning**
- Rising SPE may indicate demand or revenue shocks.
- Rising T² may indicate supply or system-level shifts.
- Rising G₂ may indicate recurring operational instability.
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

    anomaly_count = int(results["Anomaly"].sum())
    anomaly_rate = results["Anomaly"].mean() * 100

    st.markdown(f"""
### Graph Insight

This graph converts monitoring scores into simple anomaly alarms.

**What it shows**
- Alarm = 0 means normal behaviour.
- Alarm = 1 means anomaly detected.
- Alarm is triggered when SPE, T², or G₂ crosses the selected threshold.

**AI Interpretation**
- Total anomalies detected: **{anomaly_count}**
- Anomaly rate: **{anomaly_rate:.2f}%**

**Business Meaning**
- Helps users quickly identify abnormal periods.
- Supports faster action on demand, supply, or operational issues.
- Converts technical monitoring into business-ready alerts.
""")

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
