import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="PCA Retail Monitoring",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# -----------------------------
# Helper functions
# -----------------------------
def compute_t2(scores: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """Compute Hotelling T² for each score vector."""
    lambda_inv = np.diag(1 / eigenvalues)
    t2 = np.array([
        t.reshape(1, -1) @ lambda_inv @ t.reshape(-1, 1)
        for t in scores
    ]).flatten()
    return t2


def build_contribution_df(
    residual_row: np.ndarray,
    vars_used: list[str]
) -> pd.DataFrame:
    """Build squared residual contribution table."""
    contribution = residual_row ** 2
    contrib_df = pd.DataFrame({
        "Variable": vars_used,
        "Contribution": contribution
    }).sort_values("Contribution", ascending=False)
    return contrib_df


def build_reconstruction_df(
    z: np.ndarray,
    pca: PCA,
    p_hat: np.ndarray,
    vars_used: list[str]
) -> pd.DataFrame:
    """Reconstruct one variable at a time and measure error reduction."""
    z_hat = (pca.transform(z.reshape(1, -1)) @ p_hat.T).flatten()
    base_spe = np.sum((z - z_hat) ** 2)

    rows = []
    for j, var in enumerate(vars_used):
        z_recon = z.copy()
        z_recon[j] = z_hat[j]

        z_recon_hat = (pca.transform(z_recon.reshape(1, -1)) @ p_hat.T).flatten()
        spe_recon = np.sum((z_recon - z_recon_hat) ** 2)

        rows.append({
            "Variable": var,
            "Base_SPE": base_spe,
            "Reconstructed_SPE": spe_recon,
            "Error_Reduction": base_spe - spe_recon
        })

    recon_df = pd.DataFrame(rows).sort_values("Error_Reduction", ascending=False)
    return recon_df


def plot_monitoring_charts(
    results: pd.DataFrame,
    spe_threshold: float,
    t2_threshold: float,
    g2_threshold: float
) -> plt.Figure:
    """Create the 4 stacked monitoring charts."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(results.index, results["SPE"], color="blue")
    axes[0].axhline(spe_threshold, color="red", linestyle="--")
    axes[0].set_title("SPE")

    axes[1].plot(results.index, results["T2"], color="green")
    axes[1].axhline(t2_threshold, color="red", linestyle="--")
    axes[1].set_title("Hotelling T²")

    axes[2].plot(results.index, results["G2"], color="purple")
    axes[2].axhline(g2_threshold, color="red", linestyle="--")
    axes[2].set_title("G₂ Index")

    axes[3].plot(results.index, results["ALARM"], color="black", drawstyle="steps-post")
    axes[3].set_title("Alarm Trigger (0 = Normal, 1 = Fault)")
    axes[3].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    return fig


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    palette: str
) -> plt.Figure:
    """Generic bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x=x_col, y=y_col, palette=palette, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_selected_pc(scores: np.ndarray, title: str) -> plt.Figure:
    """Plot selected PC score over monitoring window."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scores, color="teal")
    ax.set_title(title)
    ax.set_xlabel("Monitoring Index")
    ax.set_ylabel("PC Score")
    plt.tight_layout()
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("PCA Retail Monitoring")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

default_candidate_vars = [
    "sales_qty",
    "sales_revenue",
    "lead_time_days",
    "delivery_reliability",
    "obsolescence_risk"
]

nominal_pct = st.sidebar.slider(
    "Nominal data split (%)",
    min_value=50,
    max_value=90,
    value=70,
    step=5
)

threshold_pct = st.sidebar.slider(
    "Threshold percentile",
    min_value=90,
    max_value=99,
    value=99,
    step=1
)

selected_pc = st.sidebar.selectbox(
    "Select Principal Component",
    options=["PC1", "PC2", "PC3"],
    index=0
)

# -----------------------------
# Main app
# -----------------------------
st.title("PCA-Based Real-Time Monitoring for Retail Inventory")

st.markdown(
    """
This app demonstrates a **PCA-based monitoring framework** for retail inventory data.
It is designed to show three layers of outcomes:

- **Technical outcome:** real-time monitoring using **SPE, T², and G₂**
- **Analytical outcome:** anomaly interpretation using **contribution** and **reconstruction**
- **Business outcome:** actionable **inventory recommendations**
"""
)

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin.")
    st.stop()

# -----------------------------
# Load and prepare data
# -----------------------------
df = pd.read_csv(uploaded_file)
st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
candidate_vars = [c for c in default_candidate_vars if c in df.columns]

if len(candidate_vars) < 3:
    st.warning(
        "Default monitored variables were not fully found. "
        "Select at least 3 numeric columns manually."
    )
    candidate_vars = numeric_cols

vars_used = st.sidebar.multiselect(
    "Variables used for monitoring",
    options=numeric_cols,
    default=candidate_vars[:5] if len(candidate_vars) >= 3 else numeric_cols[:5]
)

if len(vars_used) < 3:
    st.error("Please select at least 3 numeric variables.")
    st.stop()

# -----------------------------
# PC-specific variable groups
# -----------------------------
pc_variable_groups = {
    "PC1": [
        "sales_qty",
        "sales_revenue",
        "margin",
        "sales_frequency"
    ],
    "PC2": [
        "lead_time_days",
        "delivery_reliability",
        "obsolescence_risk",
        "supply_variability",
        "stockout_risk"
    ],
    "PC3": [
        "inventory_turnover",
        "fill_rate",
        "service_level",
        "order_cycle_time",
        "forecast_error"
    ]
}

pc_available_variables = {
    pc: [v for v in variables if v in vars_used]
    for pc, variables in pc_variable_groups.items()
}

X = df[vars_used].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

split_idx = int(len(X) * (nominal_pct / 100))
X_nominal = X.iloc[:split_idx].copy()
X_monitor = X.iloc[split_idx:].copy()

if len(X_monitor) == 0:
    st.error("Monitoring dataset is empty. Reduce the nominal split percentage.")
    st.stop()

# -----------------------------
# Standardization + PCA
# -----------------------------
scaler = StandardScaler()
X_nominal_scaled = scaler.fit_transform(X_nominal)
X_monitor_scaled = scaler.transform(X_monitor)

A = min(3, X_nominal.shape[1])

pca = PCA(n_components=A)
T_nominal = pca.fit_transform(X_nominal_scaled)
T_monitor = pca.transform(X_monitor_scaled)

p_hat = pca.components_.T
lambda_hat = pca.explained_variance_

# Full PCA for G2
pca_full = PCA(n_components=X_nominal.shape[1])
pca_full.fit(X_nominal_scaled)

T_nominal_full = pca_full.transform(X_nominal_scaled)
T_monitor_full = pca_full.transform(X_monitor_scaled)
lambda_full = pca_full.explained_variance_
m = X_nominal.shape[1]

# -----------------------------
# Selected PC setup
# -----------------------------
pc_names = [f"PC{i+1}" for i in range(A)]

loadings_df = pd.DataFrame(
    pca.components_.T,
    index=vars_used,
    columns=pc_names
)

selected_pc_idx = int(selected_pc.replace("PC", "")) - 1

if selected_pc_idx >= A:
    st.warning(f"{selected_pc} is not available because only {A} principal components were retained.")
    selected_pc_idx = A - 1
    selected_pc = f"PC{A}"

selected_pc_variance = pca.explained_variance_ratio_[selected_pc_idx] * 100
selected_pc_scores_monitor = T_monitor[:, selected_pc_idx]

selected_pc_loadings = (
    loadings_df[[selected_pc]]
    .rename(columns={selected_pc: "Loading"})
    .sort_values("Loading", key=lambda s: s.abs(), ascending=False)
    .reset_index()
    .rename(columns={"index": "Variable"})
)

pc_meaning_map = {
    "PC1": "Sales Intensity & Profitability",
    "PC2": "Supply Volatility & Risk",
    "PC3": "Operational Stability"
}

pc_business_map = {
    "PC1": {
        "title": "Sales Intensity & Profitability",
        "meaning": "This component reflects sales performance, revenue generation, and product profitability.",
        "focus_variables": pc_available_variables.get("PC1", []),
        "issue": "Large deviation suggests abnormal sales movement, unusual profitability shifts, or demand spikes/drops.",
        "action": "Review stock allocation, pricing strategy, and replenishment planning for high-impact SKUs."
    },
    "PC2": {
        "title": "Supply Volatility & Risk",
        "meaning": "This component reflects supplier delay, delivery inconsistency, and inventory risk exposure.",
        "focus_variables": pc_available_variables.get("PC2", []),
        "issue": "Large deviation suggests supply instability, logistics disruption, or rising stockout/obsolescence risk.",
        "action": "Review lead times, supplier performance, safety stock policy, and contingency sourcing plans."
    },
    "PC3": {
        "title": "Operational Stability",
        "meaning": "This component reflects the consistency and efficiency of inventory operations and execution.",
        "focus_variables": pc_available_variables.get("PC3", []),
        "issue": "Large deviation suggests operational instability, process inefficiency, or recurring execution issues.",
        "action": "Investigate workflow consistency, planning accuracy, service levels, and operational control performance."
    }
}

pc_info = pc_business_map.get(selected_pc, {
    "title": "Selected Principal Component",
    "meaning": "This principal component captures key system variation.",
    "focus_variables": [],
    "issue": "Deviation indicates a change in behaviour that should be investigated.",
    "action": "Review the highest contributing variables and take corrective action."
})

# -----------------------------
# Residuals
# -----------------------------
X_nominal_hat = T_nominal @ p_hat.T
X_monitor_hat = T_monitor @ p_hat.T

E_nominal = X_nominal_scaled - X_nominal_hat
E_monitor = X_monitor_scaled - X_monitor_hat

# -----------------------------
# Indices
# -----------------------------
SPE_nominal = np.sum(E_nominal ** 2, axis=1)
SPE_monitor = np.sum(E_monitor ** 2, axis=1)
SPE_threshold = np.percentile(SPE_nominal, threshold_pct)

T2_nominal = compute_t2(T_nominal, lambda_hat)
T2_monitor = compute_t2(T_monitor, lambda_hat)
T2_threshold = np.percentile(T2_nominal, threshold_pct)

if m < 2:
    st.error("Need at least 2 variables to compute G₂.")
    st.stop()

idx = np.arange(m - 2, m)
G2_nominal = np.sum((T_nominal_full[:, idx] ** 2) / lambda_full[idx], axis=1)
G2_monitor = np.sum((T_monitor_full[:, idx] ** 2) / lambda_full[idx], axis=1)
G2_threshold = np.percentile(G2_nominal, threshold_pct)

# -----------------------------
# Results table
# -----------------------------
results = pd.DataFrame(index=X_monitor.index)
results["SPE"] = SPE_monitor
results["SPE_alarm"] = (results["SPE"] > SPE_threshold).astype(int)

results["T2"] = T2_monitor
results["T2_alarm"] = (results["T2"] > T2_threshold).astype(int)

results["G2"] = G2_monitor
results["G2_alarm"] = (results["G2"] > G2_threshold).astype(int)

results["ALARM"] = (
    (results["SPE_alarm"] == 1) |
    (results["T2_alarm"] == 1) |
    (results["G2_alarm"] == 1)
).astype(int)

alarm_points = results.index[results["ALARM"] == 1]
first_alarm = alarm_points[0] if len(alarm_points) > 0 else None

# -----------------------------
# Root cause tables
# -----------------------------
contrib_df = None
recon_df = None
business_df = None

if first_alarm is not None:
    row_pos = list(results.index).index(first_alarm)

    contrib_df = build_contribution_df(
        residual_row=E_monitor[row_pos],
        vars_used=vars_used
    )

    recon_df = build_reconstruction_df(
        z=X_monitor_scaled[row_pos].copy(),
        pca=pca,
        p_hat=p_hat,
        vars_used=vars_used
    )

    business_map = {
        "sales_qty": "Demand spike/drop",
        "sales_revenue": "Revenue abnormality",
        "lead_time_days": "Supplier / logistics delay",
        "delivery_reliability": "Delivery inconsistency",
        "obsolescence_risk": "Inventory risk issue"
    }

    action_map = {
        "sales_qty": "Adjust inventory levels",
        "sales_revenue": "Review pricing / sales trend",
        "lead_time_days": "Expedite orders or switch supplier",
        "delivery_reliability": "Improve delivery planning",
        "obsolescence_risk": "Reduce inventory or run promotions"
    }

    business_df = contrib_df.copy()
    business_df["Issue_Detected"] = business_df["Variable"].map(business_map).fillna("Operational anomaly")
    business_df["Business_Action"] = business_df["Variable"].map(action_map).fillna("Investigate and monitor closely")

# -----------------------------
# Tabs - 3 tabs only
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Technical Outcome",
    "Analytical Outcome",
    "Business Outcome"
])

# -----------------------------
# TAB 1: Technical Outcome
# -----------------------------
with tab1:
    st.subheader("Technical Outcome")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PCA Components", A)
    c2.metric("Variance Retained (%)", round(pca.explained_variance_ratio_.sum() * 100, 2))
    c3.metric("Alarms Detected", int(results["ALARM"].sum()))
    c4.metric("Variables Monitored", len(vars_used))

    st.markdown("### Monitoring Charts")
    fig_monitor = plot_monitoring_charts(
        results=results,
        spe_threshold=SPE_threshold,
        t2_threshold=T2_threshold,
        g2_threshold=G2_threshold
    )
    st.pyplot(fig_monitor)

    st.markdown("### Selected Principal Component")

    c5, c6 = st.columns([1, 2])

    with c5:
        st.metric("Selected PC", selected_pc)
        st.metric("Variance Explained (%)", round(selected_pc_variance, 2))
        st.info(f"**{selected_pc}** represents **{pc_meaning_map.get(selected_pc, 'Key system behaviour')}**.")

        st.markdown("### Variables Associated with Selected PC")
        selected_pc_vars = pc_available_variables.get(selected_pc, [])
        if selected_pc_vars:
            st.write(selected_pc_vars)
        else:
            st.warning(f"No dedicated variables from the {selected_pc} group were found in the uploaded dataset.")

    with c6:
        st.pyplot(
            plot_selected_pc(
                selected_pc_scores_monitor,
                f"{selected_pc} Score Across Monitoring Window"
            )
        )

    st.markdown("### Technical Interpretation")
    st.write(
        """
- **SPE** identifies sudden abnormal deviations in demand or supply patterns  
- **T²** detects structural changes in overall system behaviour  
- **G₂** provides robust detection of persistent and subtle anomalies  
- **Alarm** confirms anomalies when detection indices exceed thresholds
"""
    )

    if first_alarm is not None:
        st.success(f"First alarm detected at monitoring index: {first_alarm}")
    else:
        st.info("No alarms were triggered in the current monitoring data.")

# -----------------------------
# TAB 2: Analytical Outcome
# -----------------------------
with tab2:
    st.subheader("Analytical Outcome")

    st.markdown("### Selected PC Loadings")

    selected_pc_vars = pc_available_variables.get(selected_pc, [])

    if selected_pc_vars:
        selected_pc_loadings_filtered = selected_pc_loadings[
            selected_pc_loadings["Variable"].isin(selected_pc_vars)
        ].copy()

        if selected_pc_loadings_filtered.empty:
            selected_pc_loadings_filtered = selected_pc_loadings.copy()

        st.write(
            f"The table below shows the most relevant variables for **{selected_pc}** based on its business meaning."
        )

        st.dataframe(selected_pc_loadings_filtered, use_container_width=True)

        fig_pc_loadings = plot_bar(
            df=selected_pc_loadings_filtered.head(5),
            x_col="Variable",
            y_col="Loading",
            title=f"Top Variable Loadings for {selected_pc}",
            palette="viridis"
        )
        st.pyplot(fig_pc_loadings)
    else:
        st.warning(f"No matching variables found for {selected_pc}. Showing all loadings instead.")

        st.dataframe(selected_pc_loadings, use_container_width=True)

        fig_pc_loadings = plot_bar(
            df=selected_pc_loadings.head(5),
            x_col="Variable",
            y_col="Loading",
            title=f"Top Variable Loadings for {selected_pc}",
            palette="viridis"
        )
        st.pyplot(fig_pc_loadings)

    if first_alarm is None or contrib_df is None or recon_df is None:
        st.info("No anomaly detected, so analytical insights are not available.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Contribution Analysis")
            st.dataframe(contrib_df, use_container_width=True)
            fig_contrib = plot_bar(
                df=contrib_df,
                x_col="Variable",
                y_col="Contribution",
                title="Contribution Plot",
                palette="Reds_r"
            )
            st.pyplot(fig_contrib)

        with c2:
            st.markdown("### Reconstruction-Based Isolation")
            st.dataframe(recon_df, use_container_width=True)
            fig_recon = plot_bar(
                df=recon_df,
                x_col="Variable",
                y_col="Error_Reduction",
                title="Reconstruction-Based Isolation",
                palette="Blues_r"
            )
            st.pyplot(fig_recon)

        st.info(f"Most likely root-cause variable: **{contrib_df.iloc[0]['Variable']}**")

        st.markdown("### Analytical Summary")
        st.write(
            "The analytical layer explains what caused the anomaly by identifying the highest contributing variable "
            "and validating it through reconstruction-based isolation."
        )

# -----------------------------
# TAB 3: Business Outcome
# -----------------------------
with tab3:
    st.subheader("Business Outcome")

    st.markdown("### Selected PC Business Interpretation")
    st.write(f"**Meaning:** {pc_info['meaning']}")
    st.write(f"**Issue if abnormal:** {pc_info['issue']}")
    st.write(f"**Recommended action:** {pc_info['action']}")

    st.markdown("### Variables Driving This Component")
    if pc_info.get("focus_variables"):
        st.write(pc_info["focus_variables"])
    else:
        st.info("No dedicated variables from this PC group were found in the dataset.")

    if first_alarm is None or business_df is None:
        st.info("No business action required because no anomaly was detected.")
    else:
        st.dataframe(business_df, use_container_width=True)

        top_var = business_df.iloc[0]["Variable"]
        top_issue = business_df.iloc[0]["Issue_Detected"]
        top_action = business_df.iloc[0]["Business_Action"]

        st.warning(f"Primary issue detected: **{top_issue}**")
        st.success(f"Recommended action for **{top_var}**: **{top_action}**")

        st.markdown("### Business Summary")
        st.write(
            "The business layer converts anomaly detection outputs into practical inventory and supply chain actions."
        )

        st.markdown("---")
        st.markdown("## AI Solution Recommendations")

        if selected_pc == "PC1":
            recommendations = [
                "Use **AI-driven demand sensing** to detect sudden sales spikes or drops early.",
                "Apply **dynamic pricing and replenishment optimization** for high-revenue SKUs.",
                "Use **profitability-aware inventory planning** to prioritize high-margin products.",
                "Combine anomaly detection with **sales forecasting models** for proactive stock planning."
            ]
        elif selected_pc == "PC2":
            recommendations = [
                "Use **supplier risk scoring models** to identify unstable vendors early.",
                "Apply **lead-time prediction models** to improve replenishment reliability.",
                "Use **stockout risk prediction** to strengthen safety stock planning.",
                "Deploy **real-time logistics alerts** when supply behaviour becomes unstable."
            ]
        elif selected_pc == "PC3":
            recommendations = [
                "Use **process stability monitoring models** to detect recurring execution inefficiencies.",
                "Apply **service-level prediction models** to anticipate operational degradation.",
                "Use **forecast error monitoring** to improve planning accuracy.",
                "Deploy **AI-based operational dashboards** for continuous execution control."
            ]
        else:
            recommendations = [
                "Use predictive monitoring models to detect anomalies early.",
                "Apply AI-based dashboards for decision support.",
                "Integrate anomaly detection with forecasting and inventory planning."
            ]

        for rec in recommendations:
            st.markdown(f"- {rec}")

        st.download_button(
            "Download Monitoring Results",
            data=results.to_csv(index=True).encode("utf-8"),
            file_name="PCA_Monitoring_Results.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download Business Actions",
            data=business_df.to_csv(index=False).encode("utf-8"),
            file_name="PCA_Business_Actions.csv",
            mime="text/csv"
        )
