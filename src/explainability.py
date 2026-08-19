"""
Predictive Maintenance - SHAP Explainability

Purpose:
    1. Load the trained Random Forest model.
    2. Read an actual sensor record from the AI4I dataset.
    3. Make a machine-failure prediction.
    4. Use SHAP to identify which sensor readings influenced
       that prediction.
    5. Convert the SHAP results into a simple machine-health report.

Important:
    SHAP explains the model's reasoning. It does not prove
    the physical/root cause of a machine failure.
"""

import os
import joblib
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "baseline_model.pkl"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "ai4i2020.csv"
)


# ============================================================
# MODEL FEATURES
# ============================================================

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


# Original machine-type encoding
TYPE_MAPPING = {
    "L": 0,
    "M": 1,
    "H": 2
}


# Names shown to the user
DISPLAY_NAMES = {
    "Type_Encoded": "Machine Type",
    "Air temperature": "Air Temperature",
    "Process temperature": "Process Temperature",
    "Rotational speed": "Rotational Speed",
    "Torque": "Torque",
    "Tool wear": "Tool Wear",
    "Temp_Diff": "Temperature Difference",
    "Power": "Power"
}


# ============================================================
# LOAD AND PREPARE REAL SENSOR DATA
# ============================================================

def prepare_data():

    print("Loading dataset from:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # Encode machine type
    # --------------------------------------------------------

    df["Type_Encoded"] = (
        df["Type"]
        .map(TYPE_MAPPING)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Rename original AI4I columns
    # --------------------------------------------------------

    df = df.rename(columns={
        "Air temperature [K]": "Air temperature",
        "Process temperature [K]": "Process temperature",
        "Rotational speed [rpm]": "Rotational speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool wear"
    })

    # --------------------------------------------------------
    # Create engineered features
    # --------------------------------------------------------

    df["Temp_Diff"] = (
        df["Process temperature"]
        - df["Air temperature"]
    )

    df["Power"] = (
        df["Rotational speed"]
        * df["Torque"]
    )

    # --------------------------------------------------------
    # Select exactly the features expected by model
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = df[FEATURE_COLUMNS].copy()

    return df, X


# ============================================================
# SHAP VALUES
# ============================================================

def get_failure_shap_values(model, sample):

    explainer = shap.TreeExplainer(model)

    shap_output = explainer(sample)

    values = shap_output.values

    # SHAP versions/models can return:
    #
    # (samples, features, classes)
    #
    # or
    #
    # (samples, features)

    if values.ndim == 3:

        # Class 1 = machine failure
        failure_values = values[0, :, 1]

    else:

        failure_values = values[0]

    return failure_values


# ============================================================
# FORMAT SENSOR VALUE
# ============================================================

def format_sensor_value(feature, value, machine_type):

    if feature == "Type_Encoded":
        return machine_type

    if feature == "Air temperature":
        return f"{value:.2f} K"

    if feature == "Process temperature":
        return f"{value:.2f} K"

    if feature == "Rotational speed":
        return f"{value:.0f} RPM"

    if feature == "Torque":
        return f"{value:.2f} Nm"

    if feature == "Tool wear":
        return f"{value:.0f} min"

    if feature == "Temp_Diff":
        return f"{value:.2f} K"

    if feature == "Power":
        return f"{value:.2f}"

    return f"{value:.2f}"


# ============================================================
# CREATE HUMAN-READABLE EXPLANATION
# ============================================================

def create_explanation(
    sample,
    shap_values,
    prediction,
    failure_probability,
    machine_type
):

    explanation = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Value": sample.iloc[0].values,
        "SHAP": shap_values
    })

    # Magnitude = how strongly feature influenced prediction
    explanation["Importance"] = explanation["SHAP"].abs()

    # Most influential features first
    explanation = explanation.sort_values(
        "Importance",
        ascending=False
    )

    print("\n")
    print("=" * 70)
    print("          PREDICTIVE MAINTENANCE - MACHINE HEALTH REPORT")
    print("=" * 70)

    # ========================================================
    # MACHINE STATUS
    # ========================================================

    print("\nMACHINE STATUS")
    print("-" * 70)

    if prediction == 1:

        status = "WARNING - MACHINE FAILURE PREDICTED"

        print(f"Status              : {status}")
        print(
            f"Failure Probability : "
            f"{failure_probability * 100:.2f}%"
        )

    else:

        status = "SAFE - NO FAILURE PREDICTED"

        print(f"Status              : {status}")
        print(
            f"Failure Probability : "
            f"{failure_probability * 100:.2f}%"
        )

    # ========================================================
    # SENSOR READINGS
    # ========================================================

    print("\nSENSOR READINGS")
    print("-" * 70)

    for feature in FEATURE_COLUMNS:

        value = sample.iloc[0][feature]

        print(
            f"{DISPLAY_NAMES[feature]:<22}: "
            f"{format_sensor_value(feature, value, machine_type)}"
        )

    # ========================================================
    # WHY?
    # ========================================================

    print("\nWHY DID THE MODEL MAKE THIS PREDICTION?")
    print("-" * 70)

    if prediction == 1:

        print(
            "\nThe model predicts a MACHINE FAILURE."
        )

        print(
            "The following sensor readings contributed "
            "toward the failure prediction:"
        )

        # Features that increased failure risk
        risk_features = explanation[
            explanation["SHAP"] > 0
        ].head(5)

        if len(risk_features) == 0:

            print(
                "\nNo individual feature had a positive "
                "SHAP contribution for this prediction."
            )

        else:

            for i, (_, row) in enumerate(
                risk_features.iterrows(),
                start=1
            ):

                feature = row["Feature"]
                value = row["Value"]
                shap_value = row["SHAP"]

                print(
                    f"\n{i}. {DISPLAY_NAMES[feature]}: "
                    f"{format_sensor_value(feature, value, machine_type)}"
                )

                print(
                    f"   → This reading increased the "
                    f"model's estimated failure risk."
                )

                print(
                    f"   → SHAP contribution: "
                    f"{shap_value:.4f}"
                )

        # ----------------------------------------------------
        # Factors reducing risk
        # ----------------------------------------------------

        protective_features = explanation[
            explanation["SHAP"] < 0
        ].head(3)

        if len(protective_features) > 0:

            print(
                "\nFactors that reduced the predicted "
                "failure risk:"
            )

            for _, row in protective_features.iterrows():

                feature = row["Feature"]
                value = row["Value"]

                print(
                    f"  • {DISPLAY_NAMES[feature]}: "
                    f"{format_sensor_value(feature, value, machine_type)}"
                )

    else:

        print(
            "\nThe model predicts that this machine is "
            "currently SAFE."
        )

        print(
            "\nThe strongest factors supporting the SAFE "
            "prediction are:"
        )

        # Negative SHAP = pushed prediction away from failure
        protective_features = explanation[
            explanation["SHAP"] < 0
        ].head(5)

        for i, (_, row) in enumerate(
            protective_features.iterrows(),
            start=1
        ):

            feature = row["Feature"]
            value = row["Value"]
            shap_value = row["SHAP"]

            print(
                f"\n{i}. {DISPLAY_NAMES[feature]}: "
                f"{format_sensor_value(feature, value, machine_type)}"
            )

            print(
                "   → This reading contributed toward "
                "LOWER failure risk."
            )

            print(
                f"   → SHAP contribution: "
                f"{shap_value:.4f}"
            )

        # ----------------------------------------------------
        # Factors increasing risk even though prediction is safe
        # ----------------------------------------------------

        risk_features = explanation[
            explanation["SHAP"] > 0
        ].head(3)

        if len(risk_features) > 0:

            print(
                "\nFACTORS REQUIRING ATTENTION"
            )

            print("-" * 70)

            for _, row in risk_features.iterrows():

                feature = row["Feature"]
                value = row["Value"]
                shap_value = row["SHAP"]

                print(
                    f"\n{DISPLAY_NAMES[feature]}: "
                    f"{format_sensor_value(feature, value, machine_type)}"
                )

                print(
                    "→ This reading is contributing "
                    "slightly toward higher failure risk."
                )

                print(
                    f"→ SHAP contribution: "
                    f"{shap_value:.4f}"
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("FINAL EXPLANATION")
    print("-" * 70)

    if prediction == 1:

        print(
            "\nThe machine is currently predicted to have "
            "a high risk of failure."
        )

        strongest = explanation.head(3)

        print(
            "The prediction was mainly influenced by:"
        )

        for _, row in strongest.iterrows():

            direction = (
                "increased"
                if row["SHAP"] > 0
                else "decreased"
            )

            print(
                f"  • {DISPLAY_NAMES[row['Feature']]} "
                f"→ {direction} the predicted failure risk."
            )

        print(
            "\nRecommended action: inspect the machine "
            "and investigate the sensor conditions "
            "identified above."
        )

    else:

        print(
            "\nThe machine is currently predicted to be SAFE."
        )

        strongest = explanation[
            explanation["SHAP"] < 0
        ].head(3)

        if len(strongest) > 0:

            print(
                "\nThe strongest factors supporting the "
                "SAFE prediction are:"
            )

            for _, row in strongest.iterrows():

                print(
                    f"  • {DISPLAY_NAMES[row['Feature']]}"
                )

        risk = explanation[
            explanation["SHAP"] > 0
        ].head(2)

        if len(risk) > 0:

            print(
                "\nHowever, the following readings are "
                "contributing toward some failure risk:"
            )

            for _, row in risk.iterrows():

                print(
                    f"  • {DISPLAY_NAMES[row['Feature']]}"
                )

    print("\n")
    print("=" * 70)
    print("                 END OF MACHINE HEALTH REPORT")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def explain_prediction():

    print("=" * 70)
    print("       PREDICTIVE MAINTENANCE - SHAP EXPLAINABILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")
    print(f"Model type: {type(model).__name__}")

    # --------------------------------------------------------
    # Load real sensor readings
    # --------------------------------------------------------

    df, X = prepare_data()

    print(
        f"\nDataset loaded successfully."
    )

    print(
        f"Number of sensor records: {len(X)}"
    )

    # --------------------------------------------------------
    # Select one REAL sensor record
    # --------------------------------------------------------

    # Change this number to explain another machine record.
    ROW_NUMBER = 0

    sample = X.iloc[[ROW_NUMBER]]

    # Get original machine type
    machine_type = df.iloc[ROW_NUMBER]["Type"]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(sample)[0]
    )

    probabilities = model.predict_proba(sample)[0]

    failure_probability = float(
        probabilities[1]
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print("\nGenerating SHAP explanation...")

    shap_values = get_failure_shap_values(
        model,
        sample
    )

    # --------------------------------------------------------
    # Generate final report
    # --------------------------------------------------------

    create_explanation(
        sample,
        shap_values,
        prediction,
        failure_probability,
        machine_type
    )


if __name__ == "__main__":
    explain_prediction()