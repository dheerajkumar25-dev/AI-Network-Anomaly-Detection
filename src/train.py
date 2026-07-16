# train.py
# Trains a model on the processed data, evaluates it, and saves:
#   - the model (models/model.pkl, always the latest)
#   - a timestamped copy (models/versions/) for version history
#   - a metrics report + two plots
#
# Uses XGBoost by default. If xgboost isn't installed, it automatically
# falls back to Random Forest so the pipeline still runs end to end.
#
# Run from the project root (after preprocess.py):
#   python src/train.py

import os
import json
import datetime
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

try:
    from xgboost import XGBClassifier
    MODEL_NAME = "XGBoost"
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    MODEL_NAME = "RandomForest (fallback - run 'pip install xgboost' to use XGBoost)"


def build_model():
    if MODEL_NAME == "XGBoost":
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


def main():
    print(f"Model type: {MODEL_NAME}")

    df = pd.read_csv("data/processed/processed_traffic.csv")
    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    os.makedirs("reports/images", exist_ok=True)
    os.makedirs("models/versions", exist_ok=True)

    with open("reports/metrics.txt", "w") as f:
        f.write("AI Network Anomaly Detection - Model Evaluation\n")
        f.write("=" * 48 + "\n\n")
        f.write(f"Model type: {MODEL_NAME}\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["normal", "anomaly"])
    plt.yticks([0, 1], ["normal", "anomaly"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("reports/images/confusion_matrix.png")
    plt.close()

    # Feature importance plot
    importance = model.feature_importances_
    order = importance.argsort()[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(FEATURES)), importance[order])
    plt.xticks(range(len(FEATURES)), [FEATURES[i] for i in order], rotation=45, ha="right")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("reports/images/feature_importance.png")
    plt.close()

    # Save the model that the app actually uses
    joblib.dump(model, "models/model.pkl")

    # Also keep a timestamped copy + a history log, so you can point to
    # "model versioning" as a real feature in interviews
    version_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, f"models/versions/model_{version_id}.pkl")

    history_path = "models/version_history.json"
    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    history.append({
        "version": version_id,
        "model_type": MODEL_NAME,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    })
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Model saved to models/model.pkl (version {version_id})")
    print("Metrics saved to reports/metrics.txt")


if __name__ == "__main__":
    main()
