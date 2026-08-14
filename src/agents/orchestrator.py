from agents.monitoring_agent import monitoring_agent
from agents.diagnosis_agent import diagnosis_agent
from agents.recommendation_agent import recommendation_agent

def run_pipeline(row: dict) -> dict:
    """
    Runs one sensor row through the full agent pipeline.
    Returns None if no anomaly detected (nothing to report).
    """
    is_anomaly = monitoring_agent(row)

    if not is_anomaly:
        return {"status": "normal", "action": "No action needed"}

    diagnosis = diagnosis_agent(row)
    recommendation = recommendation_agent(diagnosis)

    return {
        "status": "anomaly_detected",
        "diagnosis": diagnosis,
        "recommendation": recommendation
    }


# Quick manual test
if __name__ == "__main__":
    sample_row = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0
    }

    result = run_pipeline(sample_row)
    print(result)

# quick test, paste this temporarily at the bottom of orchestrator.py or run in a scratch script
from agents.diagnosis_agent import diagnosis_agent

risky_row = {
    "Type": "L",
    "Air temperature [K]": 300.5,
    "Process temperature [K]": 311.2,
    "Rotational speed [rpm]": 1350,
    "Torque [Nm]": 65.0,
    "Tool wear [min]": 220
}
print(diagnosis_agent(risky_row))