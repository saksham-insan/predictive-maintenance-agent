import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from anomaly_detection import is_anomalous, DEFAULT_MODEL_PATH

model = joblib.load(DEFAULT_MODEL_PATH)

def monitoring_agent(row: dict) -> bool:
    """
    Checks if a sensor row looks anomalous using the trained Isolation Forest.
    """
    return is_anomalous(model, row)