"""
Geetika's Failure Prediction Model
Predictive Maintenance Agent

This module trains an Extra Trees Classifier on the
AI4I 2020 Predictive Maintenance Dataset.

The model predicts whether a machine will fail:

0 = No Failure
1 = Failure
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "ai4i2020.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "geetika_extra_trees_model.pkl"
)


# ---------------------------------------------------------
# FEATURE CONFIGURATION
# ---------------------------------------------------------

TYPE_MAPPING = {
    "L": 0,
    "M": 1,
    "H": 2
}


FEATURE_COLUMNS = [
    "Type_Encoded",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Temp_Diff",
    "Power"
]

TARGET_COLUMN = "Machine failure"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data(filepath=DATA_PATH):

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}"
        )

    df = pd.read_csv(filepath)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


# ---------------------------------------------------------
# PREPROCESS DATA
# ---------------------------------------------------------

def preprocess_data(df):

    data = df.copy()

    # Encode Type
    if "Type" in data.columns:
        data["Type_Encoded"] = (
            data["Type"]
            .map(TYPE_MAPPING)
            .fillna(0)
            .astype(int)
        )

    # Rename columns
    rename_cols = {
        "Air temperature [K]": "Air temperature",
        "Process temperature [K]": "Process temperature",
        "Rotational speed [rpm]": "Rotational speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool wear"
    }

    data = data.rename(columns=rename_cols)

    # Temperature difference
    data["Temp_Diff"] = (
        data["Process temperature"]
        - data["Air temperature"]
    )

    # Mechanical power
    data["Power"] = (
        data["Rotational speed"]
        * data["Torque"]
    )

    # Check required columns
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    X = data[FEATURE_COLUMNS]

    y = data[TARGET_COLUMN]

    return X, y


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

def train_model(X_train, y_train):

    print("\nTraining Extra Trees Classifier...")

    model = ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("Training completed.")

    return model


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "=" * 50)
    print("EXTRA TREES MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("=" * 50)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

def save_model(model):

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved successfully at:\n{MODEL_PATH}"
    )


# ---------------------------------------------------------
# COMPLETE PIPELINE
# ---------------------------------------------------------

def run_pipeline():

    # Step 1: Load dataset
    df = load_data()

    # Step 2: Preprocess dataset
    X, y = preprocess_data(df)

    print("\nFeatures used:")
    print(X.columns.tolist())

    # Step 3: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDataset split:")
    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # Step 4: Train model
    model = train_model(
        X_train,
        y_train
    )

    # Step 5: Evaluate model
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Step 6: Save model
    save_model(model)

    return model, metrics


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    run_pipeline()
