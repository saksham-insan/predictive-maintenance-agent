"""
Business Impact Quantification for Predictive Maintenance Agent.

Scans the full dataset through the agent pipeline and reports how well
the system would have performed as an early-warning tool against real
historical failures.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import pandas as pd
from agents.orchestrator import run_pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")

CONFIDENCE_THRESHOLD = 0.70  # what counts as a "high-confidence" catch


def row_to_dict(row: pd.Series) -> dict:
    return {
        "Type": row["Type"],
        "Air temperature [K]": row["Air temperature [K]"],
        "Process temperature [K]": row["Process temperature [K]"],
        "Rotational speed [rpm]": row["Rotational speed [rpm]"],
        "Torque [Nm]": row["Torque [Nm]"],
        "Tool wear [min]": row["Tool wear [min]"]
    }


def evaluate_business_impact(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path)
    real_failures = df[df["Machine failure"] == 1]

    print(f"Scanning {len(real_failures)} real failure cases through the pipeline...\n")

    caught_high_confidence = 0
    caught_any_flag = 0
    missed_entirely = 0

    for _, row in real_failures.iterrows():
        sensor_row = row_to_dict(row)
        result = run_pipeline(sensor_row)

        if result["status"] == "anomaly_detected":
            caught_any_flag += 1
            if result["diagnosis"]["confidence"] >= CONFIDENCE_THRESHOLD:
                caught_high_confidence += 1
        else:
            missed_entirely += 1

    total = len(real_failures)
    print("=" * 55)
    print("BUSINESS IMPACT SUMMARY")
    print("=" * 55)
    print(f"Total real failures in dataset: {total}")
    print(f"Flagged by monitoring agent (any): {caught_any_flag} ({caught_any_flag/total*100:.1f}%)")
    print(f"Caught with HIGH confidence (>={CONFIDENCE_THRESHOLD*100:.0f}%): {caught_high_confidence} ({caught_high_confidence/total*100:.1f}%)")
    print(f"Missed entirely (no anomaly flag at all): {missed_entirely} ({missed_entirely/total*100:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    evaluate_business_impact()