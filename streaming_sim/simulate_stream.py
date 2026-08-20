"""
Streaming Simulation for Predictive Maintenance Agent.

Replays the AI4I 2020 dataset row-by-row with a delay, simulating a live
sensor feed. Each row is passed through the full agent pipeline
(monitoring -> diagnosis -> recommendation).
"""

import os
import sys
import time
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from agents.orchestrator import run_pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")


def row_to_dict(row: pd.Series) -> dict:
    return {
        "Type": row["Type"],
        "Air temperature [K]": row["Air temperature [K]"],
        "Process temperature [K]": row["Process temperature [K]"],
        "Rotational speed [rpm]": row["Rotational speed [rpm]"],
        "Torque [Nm]": row["Torque [Nm]"],
        "Tool wear [min]": row["Tool wear [min]"]
    }


def simulate_stream(data_path: str = DATA_PATH, delay_seconds: float = 0.5,
                     max_rows: int = 30, only_show_anomalies: bool = False):
    df = pd.read_csv(data_path)
    if max_rows:
        df = df.head(max_rows)

    print(f"[STREAM] Starting simulation — {len(df)} rows, {delay_seconds}s delay between readings\n")

    for i, row in df.iterrows():
        sensor_row = row_to_dict(row)
        result = run_pipeline(sensor_row)

        timestamp = f"[t={i}]"

        if result["status"] == "normal":
            if not only_show_anomalies:
                print(f"{timestamp} Reading OK — no anomaly detected")
        else:
            diagnosis = result["diagnosis"]
            recommendation = result["recommendation"]
            print(f"\n{timestamp} ⚠ ANOMALY DETECTED")
            print(f"  Prediction: {'FAILURE RISK' if diagnosis['prediction'] == 1 else 'No failure predicted'}")
            print(f"  Confidence: {diagnosis['confidence']:.0%}")
            print(f"  Why: {diagnosis.get('plain_explanation', diagnosis['explanation'])}")
            print(f"  Recommended action: {recommendation['action']} (Urgency: {recommendation['urgency']})\n")

        time.sleep(delay_seconds)

    print("[STREAM] Simulation complete.")


if __name__ == "__main__":
    simulate_stream(max_rows=30, delay_seconds=0.5, only_show_anomalies=False)