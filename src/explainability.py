from pathlib import Path

import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. DATASET PATH
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"

TARGET_COLUMN = "Machine failure"

NON_FEATURE_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the CSV and does the bare minimum cleaning/encoding needed to train a model.
    """
    print("Loading dataset...")
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=[TARGET_COLUMN] + NON_FEATURE_COLUMNS, errors="ignore")
    y = df[TARGET_COLUMN]

    if "Type" in X.columns:
        X["Type"] = X["Type"].map({"L": 0, "M": 1, "H": 2})

    valid_rows = X.notna().all(axis=1)
    X, y = X.loc[valid_rows], y.loc[valid_rows]

    print(f"Features used by the model: {list(X.columns)}")
    return X, y


def train_dummy_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a throwaway classifier just so SHAP has something to explain.
    Kept for standalone demo/testing purposes — the real pipeline uses
    diagnosis_agent.py, which calls get_plain_english_explanation() below
    against Person 2's actual trained model instead of this dummy one.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def get_failure_class_shap_values(shap_values, sample_index: int = 0):
    """
    Extracts the SHAP values for the "failure" class (class 1) from whatever
    shape shap.TreeExplainer(...).shap_values() returns.
    """
    if isinstance(shap_values, list):
        return shap_values[1][sample_index]
    elif len(shap_values.shape) == 3:
        return shap_values[sample_index, :, 1]
    else:
        return shap_values[sample_index]


def explain_one_prediction(model: RandomForestClassifier, X_background: pd.DataFrame, row_to_explain: pd.DataFrame) -> None:
    """
    Runs SHAP on a single row and prints a plain-English explanation (console demo version).
    """
    print("\nRunning SHAP...")

    explainer = shap.TreeExplainer(model, X_background)
    shap_values = explainer.shap_values(row_to_explain)
    failure_shap = get_failure_class_shap_values(shap_values)

    predicted_probability = model.predict_proba(row_to_explain)[0][1]
    prediction = model.predict(row_to_explain)[0]

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print("MACHINE FAILURE" if prediction == 1 else "NO MACHINE FAILURE")
    print(f"Predicted failure probability: {predicted_probability:.1%}")

    explanation = pd.DataFrame({
        "Feature": row_to_explain.columns,
        "Value": row_to_explain.iloc[0].values,
        "SHAP Value": failure_shap,
    })
    explanation["Impact"] = explanation["SHAP Value"].abs()
    explanation = explanation.sort_values(by="Impact", ascending=False)

    print("\n" + "=" * 60)
    print("SHAP EXPLANATION (all features, biggest impact first)")
    print("=" * 60)
    print(explanation[["Feature", "Value", "SHAP Value"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("PLAIN ENGLISH EXPLANATION (top factors)")
    print("=" * 60)
    for _, row in explanation.head(5).iterrows():
        direction = "increased" if row["SHAP Value"] > 0 else "decreased"
        print(f"- {row['Feature']} {direction} the model's tendency toward "
              f"MACHINE FAILURE (SHAP = {row['SHAP Value']:+.4f})")
    print("=" * 60)


def get_plain_english_explanation(model, background_sample: pd.DataFrame,
                                    row_to_explain: pd.DataFrame, top_n: int = 3) -> str:
    """
    Returns a short plain-English SHAP explanation string for a single prediction.
    Used by diagnosis_agent.py in the real pipeline (as opposed to
    explain_one_prediction() above, which is the console demo version).
    """
    explainer = shap.TreeExplainer(model, background_sample)
    shap_values = explainer.shap_values(row_to_explain)
    failure_shap = get_failure_class_shap_values(shap_values)

    explanation_df = pd.DataFrame({
        "Feature": row_to_explain.columns,
        "SHAP Value": failure_shap,
    })
    explanation_df["Impact"] = explanation_df["SHAP Value"].abs()
    explanation_df = explanation_df.sort_values(by="Impact", ascending=False)

    parts = []
    for _, row in explanation_df.head(top_n).iterrows():
        direction = "increased" if row["SHAP Value"] > 0 else "decreased"
        parts.append(f"{row['Feature']} {direction} failure risk")

    return "; ".join(parts)


def main():
    """Standalone demo — trains a throwaway model and explains one test row.
    Not used by the real pipeline, kept for reference/testing SHAP in isolation."""
    X, y = load_and_prepare_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\nTraining temporary Random Forest...")
    model = train_dummy_model(X_train, y_train)
    print(f"Model trained. Test accuracy: {model.score(X_test, y_test):.1%}")

    row_to_explain = X_test.iloc[[0]]
    background_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)

    explain_one_prediction(model, background_sample, row_to_explain)


if __name__ == "__main__":
    main()