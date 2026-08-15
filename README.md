# Predictive Maintenance Agent

An agentic AI pipeline that monitors industrial machine sensor data, detects early
warning signs of failure, diagnoses the risk with an explainable model, and
recommends maintenance actions — built for the Cognizant hackathon.

## Problem

Factories typically run maintenance reactively (fix after breakdown) or on a fixed
schedule (waste money servicing healthy machines). This project predicts failures
*before* they happen, using live sensor readings, so maintenance can be scheduled
proactively.

## How it works

A 4-stage agent pipeline processes each incoming sensor reading:

1. **Monitoring Agent** — Isolation Forest (unsupervised) flags statistically
   abnormal sensor readings
2. **Diagnosis Agent** — a tuned Random Forest classifier predicts failure risk,
   explained in plain English using real SHAP values (not a black box)
3. **Recommendation Agent** — converts the diagnosis into an actionable
   maintenance recommendation with urgency level
4. **Orchestrator** — chains the above into one pipeline; a **Streaming
   Simulation** replays sensor data live to demonstrate real-time monitoring,
   visualized in a live **Streamlit Dashboard**

## Key results

- **Anomaly detection**: tuned to catch 55.5% of real historical failures
  while flagging only 15% of all readings (avoiding alert fatigue)
- **Failure prediction**: 91.8% precision, 82.4% recall (tuned classification
  threshold of 0.4, chosen over both the default 0.5 and SMOTE oversampling
  after comparing precision/recall trade-offs — see `docs/architecture.md`)
- **Business impact**: the full pipeline flags ~49% of real historical
  failures with high confidence (≥70%), giving maintenance teams advance
  warning instead of reactive breakdown response

## Tech stack

Python · scikit-learn · XGBoost-adjacent (Random Forest) · SHAP · imbalanced-learn ·
Streamlit · pandas

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
— 10,000 rows of machine sensor readings (temperature, rotational speed,
torque, tool wear) with real failure labels.

## Project structure
predictive-maintenance-agent/
├── data/raw/ # AI4I 2020 dataset
├── src/
│ ├── data_pipeline.py # loading, cleaning
│ ├── anomaly_detection.py # Isolation Forest, tuned
│ ├── failure_prediction.py # Random Forest, cross-validated, threshold-tuned
│ ├── explainability.py # SHAP-based explanations
│ ├── business_impact.py # real-failure catch-rate analysis
│ └── agents/
│ ├── monitoring_agent.py
│ ├── diagnosis_agent.py
│ ├── recommendation_agent.py
│ └── orchestrator.py # chains the agents together
├── streaming_sim/ # live sensor feed simulation
├── dashboard/app.py # Streamlit live dashboard
├── models/ # trained model artifacts
└── docs/architecture.md # design + tuning decisions

## How to run

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Train models (if not already present in models/)
python src/anomaly_detection.py
python src/failure_prediction.py

# Run the live dashboard
python -m streamlit run dashboard/app.py

# Or run the streaming simulation in the terminal
python streaming_sim/simulate_stream.py
```

## Team

[list your 5-6 active members here]

## Notes

Full design decisions and tuning experiments (SMOTE vs class-weighting,
threshold tuning, anomaly contamination tuning) are documented in
`docs/architecture.md`.
