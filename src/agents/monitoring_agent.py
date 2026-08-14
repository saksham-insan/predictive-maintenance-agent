import random

def monitoring_agent(row: dict) -> bool:
    """
    Checks if a sensor row looks anomalous.
    PLACEHOLDER: returns random True/False for now.
    Replace this with Person 1's real Isolation Forest check once ready.
    """
    # TODO: replace with real anomaly_detection.py function
    return random.random() < 0.15  # ~15% of rows flagged, roughly mimics real failure rate