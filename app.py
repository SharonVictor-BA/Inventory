import io
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="PCA Retail Monitoring",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# Helper functions
# -----------------------------
def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    valid = []
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            valid.append(col)
    return valid


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame, selected_cols: List[str]) -> pd.DataFrame:
    data = df[selected_cols].copy()
    for col in selected_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.fillna(data.median(numeric_only=True))
    return data


def pca_monitoring_pipeline(
    data: pd.DataFrame,
    n_components: int = 3,
    nominal_ratio: float = 0.7,
    threshold_pct: float = 99.0,
) -> dict:
    if data.shape[0] < 20:
        raise ValueError("At least 20 rows are recommended for monitoring.")

    split_idx = int(len(data) * nominal_ratio)
    split_idx = max(10, min(split_idx, len(data) - 5))

    nominal = data.iloc[:split_idx].copy()
    monitor = data.iloc[split_idx:].copy()

    scaler = StandardScaler()
    nominal_scaled = scaler.fit_transform(nominal)
    monitor_scaled = scaler.transform(monitor)

    a = min(n_components, nominal.shape[1])
    pca = PCA(n_components=a)
    t_nominal = pca.fit_transform(nominal_scaled)
    t_monitor = pca.transform(monitor_scaled)

    p_hat = pca.components_.T
    lambda_hat = pca.explained_variance_

    # Full PCA for Gi/G2 family
    pca_full = PCA(n_components=nominal.shape[1])
    pca_full.fit(nominal_scaled)
    t_nominal_full = pca_full.transform(nominal_scaled)
    t_monitor_full = pca_full.transform(monitor_scaled)
    lambda_full = pca_full.explained_variance_
    m = nominal.shape[1]

    nominal_hat = t_nominal @ p_hat.T
    monitor_hat = t_monitor @ p_hat.T

    e_nominal = nominal_scaled - nominal_hat
    e_monitor = monitor_scaled - monitor_hat

    # SPE
    spe_nominal = np.sum(e_nominal**2, axis=1)
    spe_monitor = np.sum(e_monitor**2, axis=1)
    spe_threshold = np.percentile(spe_nominal, threshold_pct)

    # Hotelling T2
    lambda_inv = np.diag(1 / lambda_hat)
    t2_nominal = np.array([
        t.reshape(1, -1) @ lambda_inv @ t.reshape(-1, 1)
        for t in t_nominal
    ]).flatten()
    t2_monitor = np.array([
        t.reshape(1, -1) @ lambda_inv @ t.reshape(-1, 1)
        for t in t_monitor
    ]).flatten()
    t2_threshold = np.percentile(t2_nominal, threshold_pct)

    # G2 using last 2 PCs when possible, else last 1
    g_dim = 2 if m >= 2 else 1
    idx = np.arange(m - g_dim, m)
    g2_nominal = np.sum((t_nominal_full[:, idx] ** 2) / lambda_full[idx], axis=1)
    g2_monitor = np.sum((t_monitor_full[:, idx] ** 2) / lambda_full[idx], axis=1)
    g2_threshold = np.percentile(g2_nominal, threshold_pct)

    results = pd.DataFrame(index=monitor.index)
    results["SPE"] = spe_monitor
    results["SPE_alarm"] = (results["SPE"] > spe_threshold).astype(int)
    results["T2"] = t2_monitor
    results["T2_alarm"] = (results["T2"] > t2_threshold).astype(int)
    results["G2"] = g2_monitor
    results["G2_alarm"] = (results["G2"] > g2_threshold).astype(int)
    results["ALARM"] = (
        (results["SPE_alarm"] == 1)
        | (results["T2_alarm"] == 1)
        | (results["G2_alarm"] == 1)
    ).astype(int)

    first_alarm: Optional[int] = None
    contrib_df = pd.DataFrame()
    recon_df = pd.DataFrame()
    business_df = pd.DataFrame()

    alarm_points = results.index[results["ALARM"] == 1]
    if len(alarm_points) > 0:
        first_alarm = int(alarm_points[0])
        row_pos = list(results.index).index(first_alarm)

        contribution = e_monitor[row_pos] ** 2
        contrib_df = pd.DataFrame(
            {"Variable": list(data.columns), "Contribution": contribution}
        ).sort_values("Contribution", ascending=False)

        z = monitor_scaled[row_pos].copy()
        z_hat = (pca.transform(z.reshape(1, -1)) @ p_hat.T).flatten()
        base_spe = np.sum((z - z_hat) ** 2)
        recon_rows = []
        for j, var in enumerate(data.columns):
            z_recon = z.copy()
            z_recon[j] = z_hat[j]
            z_recon_hat = (pca.transform(z_recon.reshape(1, -1)) @ p_hat.T).flatten()
            spe_recon = np.sum((z_recon - z_recon_hat) ** 2)
            recon_rows.append([var, base_spe, spe_recon, base_spe - spe_recon])
        recon_df = pd.DataFrame(
            recon_rows,
            columns=["Variable", "Base_SPE", "Reconstructed_SPE", "Error_Reduction"],
        ).sort_values("Error_Reduction", ascending=False)

        business_map = {
            "sales_qty": "Demand spike/drop",
            "sales_revenue": "Revenue abnormality",
            "lead_time_days": "Supplier / logistics delay",
            "delivery_reliability": "Delivery inconsistency",
            "obsolescence_risk": "Inventory risk issue",
        }
        action_map = {
            "sales_qty": "Adjust inventory levels",
            "sales_revenue": "Review pricing or sales trend",
            "lead_time_days": "Expedite orders / switch supplier",
            "delivery_reliability": "Improve delivery planning",
            "obsolescence_risk": "Reduce inventory / promotions",
        }
        business_df = contrib_df.copy()
        business_df["Issue_Detected"] = business_df["Variable"].map(business_map).fillna("Operational anomaly")
        business_df["Business_Action"] = business_df["Variable"].map(action_map).fillna("Investigate and monitor")

    return {
        "nominal": nominal,
        "monitor": monitor,
        "results": results,
        "spe_threshold": spe_threshold,
        "t2_threshold": t2_threshold,
        "g2_threshold": g2_threshold,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "total_variance_pct": pca.explained_variance_ratio_.sum() * 100,
        "first_alarm": first_alarm,
        "contrib_df": contrib_df,
        "recon_df": recon_df,
        "business_df": business_df,
    }


def metric_figure(series: pd.Series, threshold: float, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=title,
            line=dict(color=color, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=[threshold] * len(series),
            mode="lines",
            name="Threshold",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Index",
        yaxis_title="Value",
        template="plotly_white",
    )
    return fig


def alarm_figure(series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line=dict(color="black", width=2, shape="hv"),
            name="Alarm",
        )
    )
    fig.update_layout(
        title="Alarm Trigger (0 = Normal, 1 = Fault)",
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Index",
        yaxis_title="Alarm",
        yaxis=dict(range=[-0.05, 1.05]),
        template="plotly_white",
        showlegend=False,
    )
    return fig


def df_download_button(df: pd.DataFrame, label: str, file_name: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=file_name, mime="text/csv")


# -----------------------------
# App UI
# -----------------------------
st.title("📈 PCA-Based Retail Monitoring")
st.caption("Upload a retail CSV to detect anomalies using PCA, SPE, Hotelling T², and G2.")

with st.sidebar:
    st.header("Configuration")
    nominal_ratio = st.slider("Nominal data ratio", 0.5, 0.9, 0.7, 0.05)
    threshold_pct = st.slider("Threshold percentile", 90, 99, 99)
    n_components = st.slider("Retained PCA components", 2, 3, 3)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

raw_df = load_csv(uploaded_file)
st.subheader("Dataset Preview")
st.dataframe(raw_df.head(), use_container_width=True)

suggested = validate_numeric_columns(
    raw_df,
    [
        "sales_qty",
        "sales_revenue",
        "lead_time_days",
        "delivery_reliability",
        "obsolescence_risk",
    ],
)

all_numeric = list(raw_df.select_dtypes(include=[np.number]).columns)
default_cols = suggested if len(suggested) >= 3 else all_numeric[:5]

selected_cols = st.multiselect(
    "Select monitored numeric variables",
    options=all_numeric,
    default=default_cols,
)

if len(selected_cols) < 3:
    st.warning("Please select at least 3 numeric columns.")
    st.stop()

clean_df = preprocess_data(raw_df, selected_cols)

run = st.button("Run PCA Monitoring", type="primary")

if run:
    try:
        outputs = pca_monitoring_pipeline(
            clean_df,
            n_components=n_components,
            nominal_ratio=nominal_ratio,
            threshold_pct=threshold_pct,
        )
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables", len(selected_cols))
    c2.metric("Retained PCs", min(n_components, len(selected_cols)))
    c3.metric("Variance Retained", f"{outputs['total_variance_pct']:.2f}%")
    c4.metric("Alarms Detected", int(outputs['results']['ALARM'].sum()))

    st.subheader("Explained Variance")
    variance_df = pd.DataFrame(
        {
            "Principal Component": [f"PC{i+1}" for i in range(len(outputs["explained_variance_ratio"]))],
            "Variance %": np.round(outputs["explained_variance_ratio"] * 100, 2),
        }
    )
    st.dataframe(variance_df, use_container_width=True)

    st.subheader("Monitoring Charts")
    st.plotly_chart(metric_figure(outputs["results"]["SPE"], outputs["spe_threshold"], "SPE", "blue"), use_container_width=True)
    st.plotly_chart(metric_figure(outputs["results"]["T2"], outputs["t2_threshold"], "Hotelling T²", "green"), use_container_width=True)
    st.plotly_chart(metric_figure(outputs["results"]["G2"], outputs["g2_threshold"], "G2", "purple"), use_container_width=True)
    st.plotly_chart(alarm_figure(outputs["results"]["ALARM"]), use_container_width=True)

    st.subheader("One-Line Graph Interpretation")
    graph_interpretation = pd.DataFrame(
        {
            "Graph": ["SPE", "T²", "G₂", "Alarm"],
            "Explanation": [
                "Identifies sudden abnormal deviations in demand or supply patterns",
                "Detects structural changes in overall system behaviour",
                "Provides robust detection of persistent and subtle anomalies",
                "Confirms anomalies when detection indices exceed thresholds",
            ],
        }
    )
    st.dataframe(graph_interpretation, use_container_width=True, hide_index=True)

    st.subheader("Monitoring Results")
    st.dataframe(outputs["results"], use_container_width=True)
    df_download_button(outputs["results"].reset_index(), "Download monitoring results", "PCA_Monitoring_Results.csv")

    if outputs["first_alarm"] is not None:
        st.success(f"First alarm detected at monitoring index: {outputs['first_alarm']}")

        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("Contribution Analysis")
            st.dataframe(outputs["contrib_df"], use_container_width=True, hide_index=True)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=outputs["contrib_df"]["Variable"],
                        y=outputs["contrib_df"]["Contribution"],
                        marker_color="#d62728",
                    )
                ]
            )
            fig.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            df_download_button(outputs["contrib_df"], "Download contribution analysis", "PCA_Contribution.csv")

        with c_right:
            st.subheader("Reconstruction Isolation")
            st.dataframe(outputs["recon_df"], use_container_width=True, hide_index=True)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=outputs["recon_df"]["Variable"],
                        y=outputs["recon_df"]["Error_Reduction"],
                        marker_color="#1f77b4",
                    )
                ]
            )
            fig.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            df_download_button(outputs["recon_df"], "Download reconstruction isolation", "PCA_Reconstruction.csv")

        st.subheader("Business Interpretation & Action")
        st.dataframe(outputs["business_df"], use_container_width=True, hide_index=True)
        df_download_button(outputs["business_df"], "Download business action table", "PCA_Business_Action.csv")
    else:
        st.info("No alarm was detected in the monitoring window.")

    st.subheader("Method Summary")
    st.markdown(
        """
        - PCA is trained on the nominal portion of the dataset.
        - New observations are monitored using **SPE**, **Hotelling T²**, and **G2**.
        - If any index exceeds its threshold, an **alarm** is triggered.
        - Contribution and reconstruction analysis help identify the likely root cause.
        """
    )
