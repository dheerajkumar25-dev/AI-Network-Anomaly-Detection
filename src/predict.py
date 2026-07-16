# predict.py
# Loads the trained model and predicts whether traffic is 'normal' or 'anomaly'.
# Also provides a simple explanation for each prediction (SHAP if installed,
# otherwise a fallback using the model's built-in feature importances).
#
# Used both as a script (demo below) and as a module by the Streamlit app.

import joblib
import pandas as pd

from utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES

model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

try:
    import shap
    explainer = shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the same encoding + scaling used during training."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    return df[FEATURES]


def predict_single(record: dict) -> dict:
    """
    record example:
    {
        "duration": 0.5, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 300, "dst_bytes": 500, "count": 5, "srv_count": 5,
        "serror_rate": 0.0, "same_srv_rate": 0.9,
    }
    Returns: {"prediction": "normal"/"anomaly", "confidence": 0.xx,
              "prob_normal": 0.xx, "prob_anomaly": 0.xx}
    """
    X = _prepare(pd.DataFrame([record]))
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label = "anomaly" if pred == 1 else "normal"
    return {
        "prediction": label,
        "confidence": round(float(proba[pred]), 4),
        "prob_normal": round(float(proba[0]), 4),
        "prob_anomaly": round(float(proba[1]), 4),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predicts on many rows at once. Returns the original df with two new columns."""
    X = _prepare(df)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    result = df.copy()
    result["prediction"] = ["anomaly" if p == 1 else "normal" for p in preds]
    result["confidence"] = [round(float(probas[i][preds[i]]), 4) for i in range(len(preds))]
    return result


def explain_single(record: dict, top_n: int = 5) -> list:
    """
    Returns the top_n features that most influenced the prediction, as a
    list of (feature_name, contribution) tuples. Positive contribution
    pushes the prediction toward 'anomaly', negative pushes toward 'normal'.

    Uses SHAP if available (accurate, per-prediction). Falls back to the
    model's global feature_importances_ scaled by how far this record's
    values are from the average, which is a rough but dependency-free
    approximation of "what mattered here".
    """
    X = _prepare(pd.DataFrame([record]))

    if SHAP_AVAILABLE:
        shap_values = explainer.shap_values(X)
        # newer shap versions can return a list per class; handle both cases
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        contributions = list(zip(FEATURES, values))
    else:
        importances = model.feature_importances_
        row = X.iloc[0].values
        # sign comes from whether this value is above/below the scaled mean (0, since we standardize)
        contributions = [
            (FEATURES[i], float(importances[i] * (1 if row[i] > 0 else -1)))
            for i in range(len(FEATURES))
        ]

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


if __name__ == "__main__":
    normal_sample = {
        "duration": 1.2, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 450, "dst_bytes": 700, "count": 3, "srv_count": 4,
        "serror_rate": 0.0, "same_srv_rate": 0.95,
    }
    attack_sample = {
        "duration": 0.01, "protocol_type": "icmp", "service": "other", "flag": "S0",
        "src_bytes": 20, "dst_bytes": 5, "count": 350, "srv_count": 300,
        "serror_rate": 0.9, "same_srv_rate": 0.1,
    }

    print("Normal sample :", predict_single(normal_sample))
    print("Attack sample :", predict_single(attack_sample))
    print("SHAP available:", SHAP_AVAILABLE)
    print("Attack sample explanation:", explain_single(attack_sample))
