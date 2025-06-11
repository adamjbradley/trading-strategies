import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="RCS ML Trading Dashboard", layout="wide")

st.title("📊 RCS ML Trading Dashboard")

# Load signal logs
log_file = "logs/signals.log"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines if l.strip()]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp", ascending=False)

    st.subheader("📡 Recent Trading Signals")
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.warning("No signal log file found.")

# Load equity curve and risk metrics
metrics_file = "logs/metrics.csv"
if os.path.exists(metrics_file):
    metrics = pd.read_csv(metrics_file)
    st.subheader("📈 Backtest Performance Metrics")
    st.write(metrics.set_index("Metric"))
else:
    st.warning("No metrics file found.")
