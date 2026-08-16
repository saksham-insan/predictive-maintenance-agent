# debug_anomaly.py — put this in src/, run directly
import joblib
from anomaly_detection import prepare_row, DEFAULT_MODEL_PATH

model = joblib.load(DEFAULT_MODEL_PATH)

low_risk_row = {
    "Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 0
}
high_risk_row = {
    "Type": "L", "Air temperature [K]": 300.5, "Process temperature [K]": 311.2,
    "Rotational speed [rpm]": 1350, "Torque [Nm]": 65.0, "Tool wear [min]": 220
}

for name, row in [("low_risk", low_risk_row), ("high_risk", high_risk_row)]:
    X = prepare_row(row)
    score = model.decision_function(X)[0]
    prediction = model.predict(X)[0]
    print(f"{name}: score={score:.4f}, prediction={prediction} (-1=anomaly, 1=normal)")