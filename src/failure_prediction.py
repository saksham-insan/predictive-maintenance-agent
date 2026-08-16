"""
Failure Prediction Module for Predictive Maintenance Agent.

Trains a baseline Random Forest Classifier on the AI4I 2020 Predictive Maintenance Dataset
to predict machine failures based on sensor features.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Default path locations
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_model.pkl")

TYPE_MAPPING = {'L': 0, 'M': 1, 'H': 2}

FEATURE_COLUMNS = [
    'Type_Encoded',
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear',
    'Temp_Diff',
    'Power'
]
TARGET_COLUMN = 'Machine failure'


def load_data(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please ensure 'ai4i2020.csv' is in 'data/raw/'."
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset from '{filepath}' with shape {df.shape}.")
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data = df.copy()

    if 'Type' in data.columns and 'Type_Encoded' not in data.columns:
        data['Type_Encoded'] = data['Type'].map(TYPE_MAPPING).fillna(0).astype(int)

    rename_cols = {
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    }
    data = data.rename(columns=rename_cols)

    if 'Process temperature' in data.columns and 'Air temperature' in data.columns:
        data['Temp_Diff'] = data['Process temperature'] - data['Air temperature']

    if 'Rotational speed' in data.columns and 'Torque' in data.columns:
        data['Power'] = data['Rotational speed'] * data['Torque']

    missing_features = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in dataset: {missing_features}")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN] if TARGET_COLUMN in data.columns else None

    return X, y, data


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42
) -> RandomForestClassifier:
    print(f"[INFO] Training Random Forest model (n_estimators={n_estimators}, random_state={random_state})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[INFO] Model training completed successfully.")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n" + "=" * 50)
    print("       MODEL EVALUATION METRICS (TEST SET)       ")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:<5} | FP: {cm[0,1]:<5}")
    print(f"FN: {cm[1,0]:<5} | TP: {cm[1,1]:<5}")
    print("\nClassification Report:")
    print(report)
    print("=" * 50 + "\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report
    }


def evaluate_with_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    print("\n" + "=" * 55)
    print(f"  {n_splits}-FOLD CROSS-VALIDATION RESULTS (with SMOTE)")
    print("=" * 55)
    for metric in scoring:
        scores = results[f'test_{metric}']
        print(f"{metric.capitalize():12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print("=" * 55 + "\n")

    return results


def tune_threshold_for_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                              model_name: str = "Model") -> pd.DataFrame:
    """
    Tests multiple probability thresholds for ANY trained model (Random Forest,
    XGBoost, grid-searched model, etc.) and reports precision/recall/f1 at each.
    Threshold tuning is applied AFTER training and works the same way regardless
    of which model produced the probabilities.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"\n{'='*55}")
    print(f"  THRESHOLD TUNING — {model_name}")
    print(f"{'='*55}")

    results = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precision = precision_score(y_test, y_pred_t, zero_division=0)
        recall = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        results.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})
        print(f"Threshold={t}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

    print(f"{'='*55}\n")
    return pd.DataFrame(results)


def compare_with_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    print(f"\n[INFO] Training XGBoost (scale_pos_weight={scale_pos_weight:.2f})...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 55)
    print("  XGBOOST vs RANDOM FOREST COMPARISON (default threshold=0.5)")
    print("=" * 55)
    print(f"XGBoost  — precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, roc_auc: {roc_auc:.3f}")
    print("=" * 55 + "\n")

    return {
        "model": xgb_model,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    base_model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    print("\n[INFO] Starting hyperparameter search (this will take a few minutes)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 55)
    print("  HYPERPARAMETER TUNING RESULTS")
    print("=" * 55)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validated F1 score: {grid_search.best_score_:.4f}")
    print("=" * 55 + "\n")

    return grid_search.best_estimator_


def save_model(model: Any, filepath: str = DEFAULT_MODEL_PATH,
               feature_names: Optional[list] = None) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Model successfully saved to '{filepath}'.")


def run_pipeline(
    data_path: str = DEFAULT_DATA_PATH,
    model_save_path: str = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    df = load_data(data_path)
    X, y, _ = preprocess_data(df)

    if y is None:
        raise ValueError("Target variable 'Machine failure' not found in dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Dataset split: Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}.")

    model = train_baseline_model(X_train, y_train, random_state=random_state)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, model_save_path, feature_names=FEATURE_COLUMNS)

    return {
        "model": model,
        "metrics": metrics,
        "feature_names": FEATURE_COLUMNS,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }


if __name__ == "__main__":
    result = run_pipeline()

    # --- Original Random Forest: threshold sweep ---
    tune_threshold_for_model(result["model"], result["X_test"], result["y_test"], "Original Random Forest")

    # --- XGBoost: default threshold, then threshold sweep ---
    xgb_result = compare_with_xgboost(
        result["X_train"], result["y_train"],
        result["X_test"], result["y_test"]
    )
    tune_threshold_for_model(xgb_result["model"], result["X_test"], result["y_test"], "XGBoost")

    # --- Grid-searched Random Forest: default threshold, then threshold sweep ---
    best_model = tune_hyperparameters(result["X_train"], result["y_train"])
    print("[INFO] Evaluating grid-searched model on test set (default threshold):")
    evaluate_model(best_model, result["X_test"], result["y_test"])
    tune_threshold_for_model(best_model, result["X_test"], result["y_test"], "Grid-Searched Random Forest")

    # --- Cross-validation comparison (SMOTE vs class_weight) ---
    df = load_data()
    X, y, _ = preprocess_data(df)
    evaluate_with_cross_validation(X, y)