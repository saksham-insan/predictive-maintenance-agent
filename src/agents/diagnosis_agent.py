import joblib
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from failure_prediction import preprocess_data, load_data, DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH
from explainability import get_plain_english_explanation

model = joblib.load(DEFAULT_MODEL_PATH)

# Build a background sample ONCE at import time (not per-call, that would be slow)
_full_df = load_data(DEFAULT_DATA_PATH)
_X_all, _, _ = preprocess_data(_full_df)
_background_sample = _X_all.sample(n=min(100, len(_X_all)), random_state=42)


def diagnosis_agent(row: dict) -> dict:
    """
    Takes a RAW sensor row, runs Person 2's exact preprocessing, predicts
    failure, and explains the prediction using real SHAP values.
    """
    df_row = pd.DataFrame([row])
    X, _, _ = preprocess_data(df_row)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    explanation = get_plain_english_explanation(model, _background_sample, X)

    return {
        "prediction": int(prediction),
        "confidence": round(float(probability), 2),
        "explanation": explanation
    }