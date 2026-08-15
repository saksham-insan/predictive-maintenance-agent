"""
Geetika's Failure Prediction Model
Predictive Maintenance Agent

This module trains an Extra Trees Classifier on the
AI4I 2020 Predictive Maintenance Dataset.

The model predicts:
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
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. FILE PATHS
# ---------------------------------------------------------

DATA_PATH = "data/raw/ai4i2020.csv"
MODEL_PATH = "models/geetika_extra_trees_model.pkl"


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

def load_data():
    """Load the AI4I 2020 dataset."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


# ---------------------------------------------------------
# 3. PREPROCESS DATA
# ---------------------------------------------------------

def preprocess_data(df):
    """Prepare features and target variable."""

    data = df.copy()

    # Encode machine Type:
    # L = 0, M = 1, H = 2
    type_mapping = {
        "L": 0,
        "M": 1,
        "H": 2
    }

    data["Type_Encoded"] = data["Type"].map(type_mapping)

    # Feature engineering
    data["Temp_Diff"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    data["Power"] = (
        data["Rotational speed [rpm]"]
        * data["Torque [Nm]"]
    )

    # Features used by the model
    features = [
        "Type_Encoded",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temp_Diff",
        "Power"
    ]

    target = "Machine failure"

    X = data[features]
    y = data[target]

    print("Preprocessing completed.")
    print("Number of features:", len(features))

    return X, y


# ---------------------------------------------------------
# 4. TRAIN EXTRA TREES MODEL
# ---------------------------------------------------------

def train_model(X_train, y_train):
    """Train Extra Trees Classifier."""

    model = ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("Extra Trees model trained successfully.")

    return model


# ---------------------------------------------------------
# 5. EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

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

    print("\n" + "=" * 50)
    print("GEETIKA - EXTRA TREES MODEL RESULTS")
    print("=" * 50)

    print(f"Accuracy :  {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall   :  {recall:.4f}")
    print(f"F1 Score :  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# ---------------------------------------------------------
# 6. SAVE MODEL
# ---------------------------------------------------------

def save_model(model):
    """Save trained model for later use."""

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved successfully to: {MODEL_PATH}"
    )


# ---------------------------------------------------------
# 7. COMPLETE PIPELINE
# ---------------------------------------------------------

def run_pipeline():

    # Load dataset
    df = load_data()

    # Preprocess
    X, y = preprocess_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Train model
    model = train_model(
        X_train,
        y_train
    )

    # Evaluate model
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save model
    save_model(model)

    return model, metrics


# ---------------------------------------------------------
# 8. RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
