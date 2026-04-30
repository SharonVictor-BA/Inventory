st.subheader("AI Recommendations for Selected Future Date")

if not selected_future_row.empty:
    row = selected_future_row.iloc[0]

    recommendations = []

    if "sales_qty" in row and row["sales_qty"] > future_df["sales_qty"].mean():
        recommendations.append("Demand is expected to be above average. Increase stock availability for high-demand SKUs.")

    if "sales_revenue" in row and row["sales_revenue"] > future_df["sales_revenue"].mean():
        recommendations.append("Revenue is expected to be strong. Prioritize high-value SKUs and avoid stockouts.")

    if "lead_time_days" in row and row["lead_time_days"] > future_df["lead_time_days"].mean():
        recommendations.append("Lead time is expected to increase. Review supplier commitments and plan replenishment earlier.")

    if "delivery_reliability" in row and row["delivery_reliability"] < future_df["delivery_reliability"].mean():
        recommendations.append("Delivery reliability is expected to weaken. Monitor logistics risk and prepare backup options.")

    if "obsolescence_risk" in row and row["obsolescence_risk"] > future_df["obsolescence_risk"].mean():
        recommendations.append("Obsolescence risk is expected to rise. Reduce excess stock or consider promotions.")

    if not selected_future_anomaly.empty and int(selected_future_anomaly["Future_Anomaly"].iloc[0]) == 1:
        recommendations.append("Future anomaly risk detected. Review demand, supply, and operational KPIs before this date.")

    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("Future KPI behaviour looks stable. Continue routine monitoring and replenishment planning.")
