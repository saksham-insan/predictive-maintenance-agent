from pathlib import Path

import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. DATASET PATH
# ---------------------------------------------------------

# Resolve the path relative to THIS FILE's location, not whatever directory
# the terminal happens to be in when you run the script. explainability.py
# lives in src/, so parents[1] is the project root.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"

TARGET_COLUMN = "Machine failure"

# Columns that must NEVER be used as model input features:
#   - UDI, Product ID        -> row identifiers, not real signal
#   - TWF, HDF, PWF, OSF, RNF -> labels for individual failure MODES.
#                                 These are only known AFTER a failure happens,
#                                 so including them would leak the answer
#                                 straight into the model (target leakage).
NON_FEATURE_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the CSV and does the bare minimum cleaning/encoding needed to train a model.

    Returns:
        X: DataFrame of features (with 'Type' encoded as a number)
        y: Series of the target (0 = no failure, 1 = failure)
    """
    print("Loading dataset...")
    # encoding='utf-8-sig' strips a hidden BOM character some CSV exports add
    # at the very start of the file (would otherwise corrupt the first column name).
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Using drop() instead of a hardcoded feature list means new sensor columns
    # added later by the data pipeline get picked up automatically, instead of
    # silently being ignored until someone remembers to update a list here.
    X = df.drop(columns=[TARGET_COLUMN] + NON_FEATURE_COLUMNS, errors="ignore")
    y = df[TARGET_COLUMN]

    # Machine learning models need numbers, not letters, so we map the
    # machine quality categories L/M/H (Low/Medium/High) to 0/1/2.
    if "Type" in X.columns:
        X["Type"] = X["Type"].map({"L": 0, "M": 1, "H": 2})

    # Drop any row with missing values in the features -- a throwaway model
    # doesn't need imputation logic, it just needs to not crash on a NaN.
    valid_rows = X.notna().all(axis=1)
    X, y = X.loc[valid_rows], y.loc[valid_rows]

    print(f"Features used by the model: {list(X.columns)}")
    return X, y


def train_dummy_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a throwaway classifier just so SHAP has something to explain.

    Two things are tuned here beyond the bare minimum, because "Machine failure"
    is rare (~3% of rows) -- a model that ignores that skew will just learn to
    always predict "no failure" and still look 97% accurate while being useless:
      - class_weight="balanced": makes mistakes on the rare failure class count
        for more during training, instead of them being drowned out.
      - n_estimators=200 / max_depth=8: a bit more capacity than a bare-minimum
        forest, without letting it overfit to noise on a 10k-row dataset.

    We use RandomForestClassifier specifically because SHAP has a fast, exact
    explainer for tree-based models (shap.TreeExplainer). Tomorrow, if Arjun's
    real model is also tree-based, this same explainer keeps working unchanged.
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

    SHAP's return shape has changed across versions, and can differ depending
    on the model/SHAP version installed on each teammate's machine:
      - Older SHAP: a list of two arrays, one per class -> shap_values[1]
      - Newer SHAP: a single array shaped (rows, features, classes)
      - Some binary setups: a single array shaped (rows, features) already
        representing the positive class

    Handling all three means this script doesn't break just because someone
    on the team has a different SHAP version installed.
    """
    if isinstance(shap_values, list):
        return shap_values[1][sample_index]
    elif len(shap_values.shape) == 3:
        return shap_values[sample_index, :, 1]
    else:
        return shap_values[sample_index]


def explain_one_prediction(model: RandomForestClassifier, X_background: pd.DataFrame, row_to_explain: pd.DataFrame) -> None:
    """
    Runs SHAP on a single row and prints a plain-English explanation.

    Args:
        model: the trained classifier
        X_background: a reference sample SHAP uses to know "what's typical"
        row_to_explain: the single row (1-row DataFrame) to explain
    """
    print("\nRunning SHAP...")

    # TreeExplainer is SHAP's fast, exact explainer built specifically for
    # tree-based models like RandomForest, XGBoost, and LightGBM. Passing a
    # background sample gets the more theoretically correct "interventional"
    # perturbation method instead of the default tree-path-dependent one --
    # cheap to do here since our dataset is small.
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

    # Build a small table, sorted by impact (absolute SHAP value) so the
    # biggest driver of the prediction is listed first.
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


def main():
    X, y = load_and_prepare_data(DATA_PATH)

    # 80% train / 20% test, stratified so the rare failure class is represented
    # proportionally in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\nTraining temporary Random Forest...")
    model = train_dummy_model(X_train, y_train)
    print(f"Model trained. Test accuracy: {model.score(X_test, y_test):.1%}")

    # Pick one row from the test set to explain (any row works -- this is
    # just proving the pipeline runs end-to-end).
    row_to_explain = X_test.iloc[[0]]

    # SHAP's background reference -- a sample, not the full training set, to
    # keep this fast.
    background_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)

    explain_one_prediction(model, background_sample, row_to_explain)


if __name__ == "__main__":
    main()
