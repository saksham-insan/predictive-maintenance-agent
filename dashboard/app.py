"""
Predictive Maintenance Agent — Live Dashboard.

Runs the streaming simulation and visualizes each sensor reading as it
passes through the agent pipeline (monitoring -> diagnosis -> recommendation).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import pandas as pd
import streamlit as st
from agents.orchestrator import run_pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")

st.set_page_config(page_title="Predictive Maintenance Agent", layout="wide")

st.title("🏭 Predictive Maintenance Agent")
st.caption("Live agentic pipeline: Monitoring → Diagnosis → Recommendation")


def row_to_dict(row: pd.Series) -> dict:
    return {
        "Type": row["Type"],
        "Air temperature [K]": row["Air temperature [K]"],
        "Process temperature [K]": row["Process temperature [K]"],
        "Rotational speed [rpm]": row["Rotational speed [rpm]"],
        "Torque [Nm]": row["Torque [Nm]"],
        "Tool wear [min]": row["Tool wear [min]"]
    }


# --- Sidebar controls ---
st.sidebar.header("Simulation Controls")
max_rows = st.sidebar.slider("Rows to stream", 10, 500, 100)
delay = st.sidebar.slider("Delay between readings (seconds)", 0.0, 2.0, 0.3)
start_button = st.sidebar.button("▶ Start Simulation")

# --- Session state to persist stats across reruns ---
if "total_scanned" not in st.session_state:
    st.session_state.total_scanned = 0
    st.session_state.total_anomalies = 0
    st.session_state.total_high_confidence = 0

# --- Layout placeholders ---
col1, col2, col3 = st.columns(3)
metric_scanned = col1.empty()
metric_anomalies = col2.empty()
metric_high_conf = col3.empty()

status_placeholder = st.empty()
log_placeholder = st.container()


def update_metrics():
    metric_scanned.metric("Readings Scanned", st.session_state.total_scanned)
    metric_anomalies.metric("Anomalies Flagged", st.session_state.total_anomalies)
    metric_high_conf.metric("High-Confidence Alerts", st.session_state.total_high_confidence)


update_metrics()

if start_button:
    df = pd.read_csv(DATA_PATH).head(max_rows)
    log_rows = []

    for i, row in df.iterrows():
        sensor_row = row_to_dict(row)
        result = run_pipeline(sensor_row)

        st.session_state.total_scanned += 1

        if result["status"] == "normal":
            status_placeholder.info(f"[t={i}] Reading OK — no anomaly detected")
        else:
            st.session_state.total_anomalies += 1
            diagnosis = result["diagnosis"]
            recommendation = result["recommendation"]

            if diagnosis["confidence"] >= 0.70 and diagnosis["prediction"] == 1:
                st.session_state.total_high_confidence += 1
                status_placeholder.error(
                    f"[t={i}] ⚠ HIGH RISK — {recommendation['action']} "
                    f"(Confidence: {diagnosis['confidence']:.0%})\n\n"
                    f"Why: {diagnosis['explanation']}"
                )
                log_rows.append({
                    "Time": i, "Status": "HIGH RISK",
                    "Confidence": f"{diagnosis['confidence']:.0%}",
                    "Action": recommendation["action"],
                    "Reason": diagnosis["explanation"]
                })
            else:
                status_placeholder.warning(
                    f"[t={i}] Anomaly flagged, low confidence "
                    f"({diagnosis['confidence']:.0%}) — no action needed"
                )

        update_metrics()
        time.sleep(delay)

    if log_rows:
        st.subheader("High-Risk Events Log")
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)

    st.success("Simulation complete.")