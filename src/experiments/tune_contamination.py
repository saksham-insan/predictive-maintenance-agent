from anomaly_detection import load_and_prepare, train_anomaly_model, ANOMALY_FEATURES

df = load_and_prepare()
failure_rows = df[df['Machine failure'] == 1]
X_failures = failure_rows[ANOMALY_FEATURES]
X_all = df[ANOMALY_FEATURES]

for c in [0.05, 0.10, 0.15, 0.20, 0.25]:
    model = train_anomaly_model(df, contamination=c)
    preds_failures = model.predict(X_failures)
    preds_all = model.predict(X_all)

    recall = (preds_failures == -1).sum() / len(failure_rows) * 100
    total_flagged = (preds_all == -1).sum()

    print(f"contamination={c}: recall={recall:.1f}%, total flagged={total_flagged} ({total_flagged/len(df)*100:.1f}% of all rows)")