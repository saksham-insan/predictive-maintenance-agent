# Agent Architecture

## Flow
1. Monitoring Agent — checks incoming sensor row for anomalies
2. Diagnosis Agent — if anomalous, predicts failure probability + explains why (SHAP)
3. Recommendation Agent — converts diagnosis into a maintenance action
4. Orchestrator — runs the above in sequence for each incoming row

## Data flow
sensor_row (dict) 
  → monitoring_agent(row) → is_anomaly (bool)
  → diagnosis_agent(row) → {prediction, confidence, explanation}
  → recommendation_agent(diagnosis) → {action, urgency}

  ## Anomaly Detection Tuning
Isolation Forest trained on raw sensor features + engineered interaction 
features (Temp_Diff, Power) to capture combined effects like high torque + 
high tool wear that raw features alone miss.

Contamination tuned via failure-recall analysis (tested 0.05 to 0.25):
- 0.05: 31.9% recall, 5% flagged
- 0.15: 55.5% recall, 15% flagged  ← chosen
- 0.25: 71.1% recall, 25% flagged

Chose 0.15 to balance catching real failures against flooding the system 
with false alarms — flagging 1 in 4 rows (as at 0.25) would make "anomalous" 
a meaningless signal in the live demo.
## Model Tuning Decisions
Tested SMOTE oversampling via 5-fold cross-validation vs class_weight="balanced":
- class_weight="balanced" (chosen): 94% precision, 74% recall
- SMOTE: 51% precision, 80% recall
Kept class_weight approach — a 6pt recall gain wasn't worth nearly halving 
precision, which would cause alert fatigue in a real deployment.

## Threshold Tuning
Tested classification thresholds 0.3-0.7 on the failure probability output:
- 0.5 (default): 94.4% precision, 75.0% recall, F1=0.836
- 0.4 (chosen): 91.8% precision, 82.4% recall, F1=0.868
Chose 0.4 — best F1 score, and a meaningful recall gain (+7.4pts) for only 
a small precision cost, without SMOTE's much larger precision trade-off.