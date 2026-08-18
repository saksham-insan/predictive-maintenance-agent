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
## Algorithm Comparison
Compared Random Forest (chosen) against XGBoost on the same train/test split:
- Random Forest (threshold=0.4): 91.8% precision, 82.4% recall, F1=0.868
- XGBoost (threshold=0.5): 84.6% precision, 80.9% recall, F1=0.827, ROC-AUC=0.986

XGBoost had a slightly better ROC-AUC (better class separation overall) but 
Random Forest gave a better F1 at our chosen operating threshold — kept 
Random Forest as the production model.
## Hyperparameter Tuning
Ran GridSearchCV (81 combinations, 5-fold CV) on Random Forest hyperparameters.
Best found: max_depth=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200
— cross-validated F1=0.826, but on the held-out test set (default threshold):
precision=84.9%, recall=82.4%, F1=0.836.

This did NOT beat our original model combined with the tuned 0.4 classification
threshold (precision=91.8%, recall=82.4%, F1=0.868). Kept the original model —
threshold tuning provided a bigger, cheaper improvement than hyperparameter
search in this case.

## Full Model Comparison (Fair, Threshold-Swept)
Tested 3 models across 5 classification thresholds each (15 combinations total):

| Model | Best Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest (chosen) | 0.4 | 91.8% | 82.4% | 0.868 |
| XGBoost | 0.6 | 85.9% | 80.9% | 0.833 |
| Grid-Searched Random Forest | 0.5 | 84.8% | 82.4% | 0.836 |

Original Random Forest with a tuned threshold outperformed both a different 
algorithm (XGBoost) and a hyperparameter-optimized version of itself — 
confirming it as the genuinely best choice, not just the first one tried.
## Anomaly Detection Algorithm Comparison
Compared Isolation Forest against Local Outlier Factor (LOF) and an ensemble
of both, across contamination levels 0.05-0.25:
- Isolation Forest alone (chosen, contamination=0.15): 55.5% recall, 15% flagged
- LOF alone: consistently worse than Isolation Forest at every contamination
  level tested (e.g. 34.5% recall at contamination=0.15)
- Ensemble (either model flags): 65.8% recall, but 26.3% of all rows flagged

Kept Isolation Forest alone — the ensemble's recall gain didn't justify
nearly doubling the false-alarm rate.
## Feature Importance
Random Forest's built-in feature importance confirms our SHAP explanations:
Power (24.5%), Rotational Speed (19.7%), and Torque (19.2%) are the top three
drivers of failure predictions — consistent across two independent
explainability methods (global importance and per-row SHAP).