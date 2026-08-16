import joblib
from anomaly_detection import load_and_prepare, ANOMALY_FEATURES, DEFAULT_MODEL_PATH

model = joblib.load(DEFAULT_MODEL_PATH)

df = load_and_prepare()
failure_rows = df[df['Machine failure'] == 1]
X_failures = failure_rows[ANOMALY_FEATURES]

scores = model.decision_function(X_failures)
predictions = model.predict(X_failures)

n_caught = (predictions == -1).sum()
print(f"Of {len(failure_rows)} REAL failure rows:")
print(f"  Flagged as anomalous: {n_caught} ({n_caught/len(failure_rows)*100:.1f}%)")
print(f"  Average anomaly score: {scores.mean():.4f}")