# PCA-Based Retail Monitoring Streamlit App

A Streamlit app for retail anomaly detection and inventory decision support using Principal Component Analysis (PCA).

## Features

- Upload a retail CSV file
- Select monitored numeric variables
- Train PCA on nominal data only
- Detect anomalies using:
  - SPE (Squared Prediction Error)
  - Hotelling T²
  - G2 index
- Trigger alarms when thresholds are exceeded
- Run contribution analysis for likely root cause
- Run reconstruction-based isolation
- Generate business interpretation and action table
- Download outputs as CSV

## Recommended columns

The app works best when the dataset contains columns such as:

- `sales_qty`
- `sales_revenue`
- `lead_time_days`
- `delivery_reliability`
- `obsolescence_risk`

It can also work with any other numeric columns you choose.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to Streamlit Community Cloud
3. Select this repo
4. Set the main file as `app.py`
5. Deploy

## Repository structure

```text
pca_streamlit_repo/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Output files

The app allows CSV downloads for:

- Monitoring results
- Contribution analysis
- Reconstruction isolation
- Business action table

## Project summary

This app transforms PCA from a dimensionality reduction tool into a real-time monitoring and decision-support system for retail inventory optimization.
