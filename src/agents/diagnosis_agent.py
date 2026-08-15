import joblib
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from failure_prediction import preprocess_data, load_data, DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH
from explainability import get_plain_english_explanation

model = joblib.load(DEFAULT_MODEL_PATH)

# Tuned via threshold analysis (see docs/architecture.md) — 0.4 gave the best
# F1 score, trading a small amount of precision for a meaningful recall gain
# over the default 0.5 cutoff.
CLASSIFICATION_THRESHOLD = 0.4

# Build a background sample ONCE at import time (not per-call, that would be slow)
_full_df = load_data(DEFAULT_DATA_PATH)
_X_all, _, _ = preprocess_data(_full_df)
_background_sample = _X_all.sample(n=min(100, len(_X_all)), random_state=42)


def diagnosis_agent(row: dict) -> dict:
    """
    Takes a RAW sensor row, runs Person 2's exact preprocessing, predicts
    failure using a tuned classification threshold, and explains the
    prediction using real SHAP values.
    """
    df_row = pd.DataFrame([row])
    X, _, _ = preprocess_data(df_row)

    probability = model.predict_proba(X)[0][1]
    prediction = int(probability >= CLASSIFICATION_THRESHOLD)
    explanation = get_plain_english_explanation(model, _background_sample, X)

    return {
        "prediction": prediction,
        "confidence": round(float(probability), 2),
        "explanation": explanation
    }