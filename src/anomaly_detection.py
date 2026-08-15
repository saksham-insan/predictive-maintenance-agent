"""
Anomaly Detection Module for Predictive Maintenance Agent.

Uses Isolation Forest (unsupervised) to detect abnormal sensor readings,
independent of the labeled failure prediction model. This catches early
warning signs that don't necessarily match historical failure patterns.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "anomaly_model.pkl")

TYPE_MAPPING = {'L': 0, 'M': 1, 'H': 2}

# Raw sensor readings + engineered interaction features (matches Person 2's
# failure prediction features, so anomaly detection can also sense combined
# effects like high torque + high wear, not just individually rare values)
ANOMALY_FEATURES = [
    'Type_Encoded',
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear',
    'Temp_Diff',
    'Power'
]


def load_and_prepare(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    rename_cols = {
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    }
    df = df.rename(columns=rename_cols)
    df['Type_Encoded'] = df['Type'].map(TYPE_MAPPING).fillna(0).astype(int)

    # Engineered interaction features
    df['Temp_Diff'] = df['Process temperature'] - df['Air temperature']
    df['Power'] = df['Rotational speed'] * df['Torque']

    return df


def train_anomaly_model(df: pd.DataFrame, contamination: float = 0.15) -> IsolationForest:
    """
    contamination = expected proportion of anomalies in the data.
    0.05 means we expect ~5% of rows to look abnormal — tune this later
    based on how many flags feel reasonable during testing.
    """
    X = df[ANOMALY_FEATURES]
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    print(f"[INFO] Isolation Forest trained on {len(X)} rows, contamination={contamination}")
    return model


def save_model(model, filepath: str = DEFAULT_MODEL_PATH):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Anomaly model saved to '{filepath}'")


def prepare_row(row: dict) -> pd.DataFrame:
    """Prepares a single raw sensor row for the anomaly model."""
    df_row = pd.DataFrame([row])
    rename_cols = {
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    }
    df_row = df_row.rename(columns=rename_cols)
    df_row['Type_Encoded'] = df_row['Type'].map(TYPE_MAPPING).fillna(0).astype(int)

    # Engineered interaction features — must match load_and_prepare exactly
    df_row['Temp_Diff'] = df_row['Process temperature'] - df_row['Air temperature']
    df_row['Power'] = df_row['Rotational speed'] * df_row['Torque']

    return df_row[ANOMALY_FEATURES]


def is_anomalous(model, row: dict) -> bool:
    """
    Returns True if the row is flagged as anomalous.
    IsolationForest.predict() returns -1 for anomalies, 1 for normal.
    """
    X = prepare_row(row)
    result = model.predict(X)[0]
    return result == -1


def run_pipeline():
    df = load_and_prepare()
    model = train_anomaly_model(df)
    save_model(model)

    # Quick sanity check: how many rows in the dataset get flagged?
    X = df[ANOMALY_FEATURES]
    predictions = model.predict(X)
    n_flagged = (predictions == -1).sum()
    print(f"[INFO] {n_flagged} of {len(df)} rows flagged as anomalous ({n_flagged/len(df)*100:.2f}%)")


if __name__ == "__main__":
    run_pipeline()