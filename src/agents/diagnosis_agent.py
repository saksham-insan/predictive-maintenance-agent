import joblib
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from failure_prediction import preprocess_data, DEFAULT_MODEL_PATH

model = joblib.load(DEFAULT_MODEL_PATH)

def diagnosis_agent(row: dict) -> dict:
    """
    Takes a RAW sensor row (same format as the original CSV columns)
    and runs it through the same preprocessing Person 2 used for training.
    """
    df_row = pd.DataFrame([row])
    X, _, _ = preprocess_data(df_row)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    # TODO: plug in Person 3's real explain_prediction() once ready
    explanation = "Explanation placeholder — swap in SHAP output here"

    return {
        "prediction": int(prediction),
        "confidence": round(float(probability), 2),
        "explanation": explanation
    }