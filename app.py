import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="PCA Retail App", layout="wide")

st.title("PCA-Based Retail Monitoring & Forecasting")

# -----------------------------
# Upload Data
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload dataset")
    st.stop()

df = pd.read_csv(uploaded_file)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

features = st.sidebar.multiselect(
    "Select Features",
    numeric_cols,
    default=numeric_cols[:5]
)

if len(features) < 3:
    st.error("Select at least 3 features")
    st.stop()

# -----------------------------
# Preprocessing
# -----------------------------
X = df[features].copy()
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA
# -----------------------------
pca = PCA(n_components=3)
pcs = pca.fit_transform(X_scaled)

pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3"])

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "PCA",
    "Monitoring",
    "Anomaly Detection"
])

# =========================================================
# TAB 1 — PREDICTION
# =========================================================
with tab1:
    st.subheader("Future Prediction using PCA")

    future_steps = st.slider("Forecast Steps", 5, 30, 10)

    pc_forecasts = {}

    for col in pc_df.columns:
        model = ARIMA(pc_df[col], order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=future_steps)
        pc_forecasts[col] = forecast

    pc_forecast_df = pd.DataFrame(pc_forecasts)

    # Reconstruct KPIs
    reconstructed_scaled = pca.inverse_transform(pc_forecast_df)
    forecast_kpis = scaler.inverse_transform(reconstructed_scaled)

    forecast_df = pd.DataFrame(forecast_kpis, columns=features)

    st.write("### Forecasted KPI Values")
    st.dataframe(forecast_df)

    # Plot
    st.write("### Forecast Trends")
    fig, ax = plt.subplots()

    for col in forecast_df.columns:
        ax.plot(forecast_df[col], label=col)

    ax.legend()
    st.pyplot(fig)

# =========================================================
# TAB 2 — PCA
# =========================================================
with tab2:
    st.subheader("PCA Analysis")

    st.write("### Explained Variance")
    variance = pca.explained_variance_ratio_

    st.write(pd.DataFrame({
        "Component": ["PC1", "PC2", "PC3"],
        "Variance": variance
    }))

    st.write("### PCA Scatter")
    fig, ax = plt.subplots()
    ax.scatter(pc_df["PC1"], pc_df["PC2"])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

# =========================================================
# TAB 3 — MONITORING
# =========================================================
with tab3:
    st.subheader("Monitoring (SPE & T²)")

    # Reconstruction
    X_hat = pca.inverse_transform(pcs)
    residual = X_scaled - X_hat

    # SPE
    spe = np.sum(residual**2, axis=1)

    # T2
    eigen_vals = pca.explained_variance_
    t2 = np.sum((pcs**2) / eigen_vals, axis=1)

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10,6))

    ax[0].plot(spe)
    ax[0].set_title("SPE")

    ax[1].plot(t2)
    ax[1].set_title("T²")

    st.pyplot(fig)

# =========================================================
# TAB 4 — ANOMALY DETECTION
# =========================================================
with tab4:
    st.subheader("Anomaly Detection")

    # Thresholds
    spe_threshold = np.percentile(spe, 95)
    t2_threshold = np.percentile(t2, 95)

    anomaly = (spe > spe_threshold) | (t2 > t2_threshold)

    results = pd.DataFrame({
        "SPE": spe,
        "T2": t2,
        "Anomaly": anomaly.astype(int)
    })

    st.write("### Results")
    st.dataframe(results)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(results["SPE"], label="SPE")
    ax.axhline(spe_threshold, color="r", linestyle="--")

    ax.plot(results["T2"], label="T2")
    ax.axhline(t2_threshold, color="g", linestyle="--")

    ax.legend()
    st.pyplot(fig)
