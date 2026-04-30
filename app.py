import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

st.title("PCA-Based Retail Inventory Forecasting and Anomaly Monitoring")

st.markdown("""
This application uses **Principal Component Analysis (PCA)** to reduce complex retail inventory data into meaningful components.
The system then uses these components for:

- **Future KPI prediction**
- **PCA-based interpretation**
- **Monitoring of abnormal behaviour**
- **Anomaly detection and business recommendations**

Although the dataset is historical, it is processed sequentially to simulate how a real-time monitoring system would behave when new data arrives.
""")

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.sidebar.success("Dataset uploaded successfully")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 3:
    st.error("Dataset must contain at least 3 numeric columns.")
    st.stop()

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
# Data Preparation
# --------------------------------------------------
X = df[features].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# PCA
# --------------------------------------------------
n_components = min(3, len(features))
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(X_scaled)

pc_names = [f"PC{i+1}" for i in range(n_components)]
pc_df = pd.DataFrame(pcs, columns=pc_names)

# --------------------------------------------------
# Forecast PCs using ARIMA
# --------------------------------------------------
pc_forecasts = {}

for col in pc_df.columns:
    try:
        model = ARIMA(pc_df[col], order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=future_steps)
        pc_forecasts[col] = forecast.values
    except Exception:
        # fallback if ARIMA fails
        last_value = pc_df[col].iloc[-1]
        pc_forecasts[col] = np.repeat(last_value, future_steps)

pc_forecast_df = pd.DataFrame(pc_forecasts)

# Convert future PCs back to original KPI space
future_scaled = pca.inverse_transform(pc_forecast_df)
future_values = scaler.inverse_transform(future_scaled)
future_df = pd.DataFrame(future_values, columns=features)

# Current KPI values
current_values = X.tail(future_steps).reset_index(drop=True)

comparison_df = pd.DataFrame()

for col in features:
    comparison_df[f"Current_{col}"] = current_values[col]
    comparison_df[f"Future_{col}"] = future_df[col]

# --------------------------------------------------
# Monitoring Metrics
# --------------------------------------------------
X_hat = pca.inverse_transform(pcs)
residual = X_scaled - X_hat

# SPE
spe = np.sum(residual ** 2, axis=1)

# T2
eigen_vals = pca.explained_variance_
t2 = np.sum((pcs ** 2) / eigen_vals, axis=1)

# G2 simplified combined index
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
# PCA Meaning
# --------------------------------------------------
pc_explanation = {
    "PC1": {
        "name": "Demand Intelligence",
        "meaning": "Captures sales intensity, demand movement, revenue behaviour, and profitability signals.",
        "business_value": "Helps identify demand spikes, drops, and revenue-impacting changes."
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
# TAB 1 — Prediction
# ==================================================
with tab1:
    st.header("Future KPI Prediction")

    st.markdown("""
This tab forecasts future KPI values using the PCA-transformed data.  
Instead of forecasting each KPI independently, the app forecasts the principal components and converts them back into KPI values.
""")

    st.subheader("Current vs Future Prediction")

    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("""
**How to read this table:**  
- `Current_` columns show recent historical KPI values  
- `Future_` columns show predicted KPI values  
- This helps compare whether future behaviour is expected to increase, decrease, or remain stable
""")

    st.subheader("Current vs Future KPI Trend")

    selected_kpi = st.selectbox("Select KPI to compare", features)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(current_values[selected_kpi].values, label=f"Current {selected_kpi}", marker="o")
    ax.plot(future_df[selected_kpi].values, label=f"Future {selected_kpi}", marker="o")
    ax.set_title(f"Current vs Future Prediction: {selected_kpi}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(selected_kpi)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.info(
        "This graph is useful because it visually compares recent historical behaviour with predicted future movement."
    )

# ==================================================
# TAB 2 — PCA
# ==================================================
with tab2:
    st.header("Principal Component Analysis")

    st.markdown("""
PCA reduces multiple correlated retail KPIs into fewer meaningful components.
This helps simplify the data while retaining the most important information.
""")

    variance_df = pd.DataFrame({
        "Principal Component": pc_names,
        "Variance Explained (%)": np.round(pca.explained_variance_ratio_ * 100, 2)
    })

    st.subheader("PCA Variance Explained")
    st.dataframe(variance_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(variance_df["Principal Component"], variance_df["Variance Explained (%)"])
    ax.set_title("Variance Explained by Principal Components")
    ax.set_ylabel("Variance Explained (%)")
    ax.grid(axis="y")
    st.pyplot(fig)

    st.info(
        "This graph is necessary because it shows how much information each principal component captures from the original dataset."
    )

    st.subheader("Explanation of Principal Components")

    for pc in pc_names:
        info = pc_explanation.get(pc)
        if info:
            st.markdown(f"""
### {pc} — {info['name']}
- **Meaning:** {info['meaning']}
- **Business Value:** {info['business_value']}
""")

    st.subheader("PCA Scatter Plot")

    if n_components >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pc_df["PC1"], pc_df["PC2"], alpha=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Projection: PC1 vs PC2")
        ax.grid(True)
        st.pyplot(fig)

        st.info(
            "This graph is useful because it shows how observations are distributed in reduced PCA space."
        )

# ==================================================
# TAB 3 — Monitoring
# ==================================================
with tab3:
    st.header("PCA-Based Monitoring")

    st.markdown("""
This tab tracks how the system behaves over time using PCA-based monitoring indices.
The monitoring signals help identify whether the retail system is behaving normally or starting to deviate.
""")

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(results["SPE"], label="SPE")
    ax[0].axhline(spe_threshold, linestyle="--", label="SPE Threshold")
    ax[0].set_title("SPE Monitoring")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(results["T2"], label="T²")
    ax[1].axhline(t2_threshold, linestyle="--", label="T² Threshold")
    ax[1].set_title("Hotelling T² Monitoring")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(results["G2"], label="G2")
    ax[2].axhline(g2_threshold, linestyle="--", label="G2 Threshold")
    ax[2].set_title("G2 Monitoring")
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Graph Explanation")

    st.markdown("""
- **SPE** detects sudden deviation from normal behaviour  
- **T²** detects structural shifts in the PCA space  
- **G2** captures persistent or subtle anomalies  
- The threshold line separates normal behaviour from abnormal behaviour  
""")

    st.info(
        "These graphs are necessary because they show when the system starts moving outside expected behaviour."
    )

# ==================================================
# TAB 4 — Anomaly Detection
# ==================================================
with tab4:
    st.header("Anomaly Detection and AI Recommendations")

    st.markdown("""
This tab converts monitoring signals into anomaly alerts.
If SPE, T², or G2 crosses its threshold, the observation is marked as abnormal.
""")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", len(results))
    c2.metric("Anomalies Detected", int(results["Anomaly"].sum()))
    c3.metric("Anomaly Rate (%)", round(results["Anomaly"].mean() * 100, 2))
    c4.metric("Threshold Percentile", threshold_percentile)

    st.subheader("Anomaly Detection Results")
    st.dataframe(results, use_container_width=True)

    st.subheader("Anomaly Alarm Graph")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(results["Anomaly"], drawstyle="steps-post")
    ax.set_title("Alarm Trigger: 0 = Normal, 1 = Anomaly")
    ax.set_xlabel("Index")
    ax.set_ylabel("Alarm")
    ax.grid(True)
    st.pyplot(fig)

    st.info(
        "This graph is necessary because it clearly shows when the system triggers an anomaly alarm."
    )

    st.subheader("AI Recommendations")

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
- Review top contributing demand and supply variables
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

    st.markdown("""
### Business Interpretation

The anomaly detection layer supports:
- early warning of demand or supply instability
- identification of abnormal KPI behaviour
- proactive inventory and supplier decisions
- improved business decision support
""")
