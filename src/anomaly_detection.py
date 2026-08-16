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
from sklearn.neighbors import LocalOutlierFactor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "anomaly_model.pkl")

TYPE_MAPPING = {'L': 0, 'M': 1, 'H': 2}

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

    df['Temp_Diff'] = df['Process temperature'] - df['Air temperature']
    df['Power'] = df['Rotational speed'] * df['Torque']

    return df


def train_anomaly_model(df: pd.DataFrame, contamination: float = 0.15) -> IsolationForest:
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

    df_row['Temp_Diff'] = df_row['Process temperature'] - df_row['Air temperature']
    df_row['Power'] = df_row['Rotational speed'] * df_row['Torque']

    return df_row[ANOMALY_FEATURES]


def is_anomalous(model, row: dict) -> bool:
    X = prepare_row(row)
    result = model.predict(X)[0]
    return result == -1


def check_recall(model, df: pd.DataFrame, model_name: str = "Model") -> float:
    """
    Generic recall check against real failure rows — works for any fitted
    model that exposes .predict() returning -1 for anomalies, 1 for normal.
    """
    failure_rows = df[df['Machine failure'] == 1]
    X_failures = failure_rows[ANOMALY_FEATURES]

    predictions = model.predict(X_failures)
    n_caught = (predictions == -1).sum()
    recall = n_caught / len(failure_rows) * 100

    print(f"{model_name}: caught {n_caught}/{len(failure_rows)} real failures ({recall:.1f}% recall)")
    return recall


def compare_with_lof(df: pd.DataFrame, contamination: float = 0.15) -> dict:
    """
    Compares Isolation Forest against Local Outlier Factor (LOF) — a different
    unsupervised anomaly detection approach based on local density rather than
    random partitioning.
    """
    X_all = df[ANOMALY_FEATURES]

    print("\n" + "=" * 55)
    print("  LOCAL OUTLIER FACTOR (LOF) vs ISOLATION FOREST")
    print("=" * 55)

    # novelty=True allows fit on the full dataset, then predict on any subset
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(X_all)

    lof_recall = check_recall(lof, df, "Local Outlier Factor")

    print("=" * 55 + "\n")
    return {"model": lof, "recall": lof_recall}


def compare_contamination_across_models(df: pd.DataFrame,
                                          contaminations: list = [0.05, 0.10, 0.15, 0.20, 0.25]) -> pd.DataFrame:
    """
    Sweeps contamination for BOTH Isolation Forest and LOF, so the comparison
    is fair across the same range of settings for each algorithm.
    """
    X_all = df[ANOMALY_FEATURES]
    failure_rows = df[df['Machine failure'] == 1]

    print("\n" + "=" * 60)
    print("  CONTAMINATION SWEEP — ISOLATION FOREST vs LOF")
    print("=" * 60)

    results = []
    for c in contaminations:
        iso = IsolationForest(n_estimators=100, contamination=c, random_state=42, n_jobs=-1)
        iso.fit(X_all)
        iso_recall = check_recall(iso, df, f"  IsolationForest (contamination={c})")

        lof = LocalOutlierFactor(n_neighbors=20, contamination=c, novelty=True)
        lof.fit(X_all)
        lof_recall = check_recall(lof, df, f"  LOF             (contamination={c})")

        results.append({
            "contamination": c,
            "isolation_forest_recall": iso_recall,
            "lof_recall": lof_recall
        })
        print()

    print("=" * 60 + "\n")
    return pd.DataFrame(results)


def try_ensemble(df: pd.DataFrame, contamination: float = 0.15):
    """
    Flags a row as anomalous if EITHER Isolation Forest OR LOF flags it.
    Different algorithms catch different patterns, so combining them can
    improve recall beyond either model alone.
    """
    X_all = df[ANOMALY_FEATURES]
    failure_rows = df[df['Machine failure'] == 1]
    X_failures = failure_rows[ANOMALY_FEATURES]

    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X_all)
    iso_preds = iso.predict(X_failures)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(X_all)
    lof_preds = lof.predict(X_failures)

    # Ensemble: anomalous if EITHER model says -1
    ensemble_preds = [(-1 if (a == -1 or b == -1) else 1) for a, b in zip(iso_preds, lof_preds)]
    n_caught = sum(1 for p in ensemble_preds if p == -1)
    recall = n_caught / len(failure_rows) * 100

    total_flagged_iso = (iso.predict(X_all) == -1).sum()
    total_flagged_lof = (lof.predict(X_all) == -1).sum()
    ensemble_all = [(-1 if (a == -1 or b == -1) else 1)
                     for a, b in zip(iso.predict(X_all), lof.predict(X_all))]
    total_flagged_ensemble = sum(1 for p in ensemble_all if p == -1)

    print("\n" + "=" * 55)
    print("  ENSEMBLE (Isolation Forest OR LOF)")
    print("=" * 55)
    print(f"Ensemble recall on real failures: {n_caught}/{len(failure_rows)} ({recall:.1f}%)")
    print(f"Total rows flagged: {total_flagged_ensemble}/{len(df)} ({total_flagged_ensemble/len(df)*100:.1f}%)")
    print(f"(vs Isolation Forest alone: {total_flagged_iso} flagged)")
    print(f"(vs LOF alone: {total_flagged_lof} flagged)")
    print("=" * 55 + "\n")

    return recall


def run_pipeline():
    df = load_and_prepare()
    model = train_anomaly_model(df)
    save_model(model)

    X = df[ANOMALY_FEATURES]
    predictions = model.predict(X)
    n_flagged = (predictions == -1).sum()
    print(f"[INFO] {n_flagged} of {len(df)} rows flagged as anomalous ({n_flagged/len(df)*100:.2f}%)")


if __name__ == "__main__":
    run_pipeline()

    df = load_and_prepare()

    # Single comparison at current contamination setting
    compare_with_lof(df)

    # Fair sweep across contamination values for both algorithms
    compare_contamination_across_models(df)

    # Does combining both algorithms catch more real failures?
    try_ensemble(df)