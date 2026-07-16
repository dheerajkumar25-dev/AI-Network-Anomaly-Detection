# preprocess.py
# Loads network traffic data, cleans it, encodes text columns to numbers,
# and scales numeric columns. Saves the ready-to-train data to data/processed/.
#
# Run from the project root:
#   python src/preprocess.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

RAW_REAL_FILE = "data/raw/KDDTrain+.txt"      # place the real NSL-KDD file here if you have it
RAW_SYNTHETIC_FILE = "data/raw/network_traffic.csv"
PROCESSED_FILE = "data/processed/processed_traffic.csv"

# Full NSL-KDD column names (41 features + label + difficulty score).
# Used only if we find the real KDDTrain+.txt file.
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level",
]


def load_real_dataset():
    """Loads the real NSL-KDD file if the user has placed it in data/raw/."""
    df = pd.read_csv(RAW_REAL_FILE, names=NSL_KDD_COLUMNS)
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["label"]].copy()
    # NSL-KDD gives ~23 specific attack names (neptune, smurf, satan, ...).
    # We simplify to the standard binary task: normal vs anomaly.
    df["label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "anomaly")
    return df


def generate_synthetic_dataset(n_normal=8000, n_anomaly=2000, seed=42):
    """
    Fallback dataset used only when the real NSL-KDD file isn't available.
    Built to have the same columns/shape as NSL-KDD so the rest of the
    pipeline doesn't need to change once you drop in the real file.
    """
    rng = np.random.default_rng(seed)

    def make_rows(n, kind):
        if kind == "normal":
            duration = rng.exponential(2.0, n)
            src_bytes = rng.normal(500, 150, n).clip(0)
            dst_bytes = rng.normal(800, 200, n).clip(0)
            count = rng.integers(1, 10, n).astype(float)
            srv_count = rng.integers(1, 10, n).astype(float)
            serror_rate = rng.uniform(0, 0.05, n)
            same_srv_rate = rng.uniform(0.7, 1.0, n)
            protocol_type = rng.choice(["tcp", "udp"], n, p=[0.8, 0.2])
            service = rng.choice(["http", "ftp", "smtp"], n, p=[0.7, 0.15, 0.15])
            flag = rng.choice(["SF"], n)
        else:
            duration = rng.exponential(0.2, n)
            src_bytes = rng.normal(50, 30, n).clip(0)
            dst_bytes = rng.normal(20, 15, n).clip(0)
            count = rng.integers(50, 500, n).astype(float)
            srv_count = rng.integers(50, 500, n).astype(float)
            serror_rate = rng.uniform(0.5, 1.0, n)
            same_srv_rate = rng.uniform(0.0, 0.3, n)
            protocol_type = rng.choice(["tcp", "udp", "icmp"], n, p=[0.5, 0.2, 0.3])
            service = rng.choice(["http", "ftp", "smtp", "other"], n, p=[0.3, 0.2, 0.2, 0.3])
            flag = rng.choice(["S0", "REJ", "RSTO"], n)

        return pd.DataFrame({
            "duration": duration, "protocol_type": protocol_type, "service": service,
            "flag": flag, "src_bytes": src_bytes, "dst_bytes": dst_bytes,
            "count": count, "srv_count": srv_count, "serror_rate": serror_rate,
            "same_srv_rate": same_srv_rate, "label": kind,
        })

    df = pd.concat([make_rows(n_normal, "normal"), make_rows(n_anomaly, "anomaly")], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add some overlap between classes so the problem isn't trivially easy
    # (real traffic is never perfectly separable).
    noisy_rows = rng.random(len(df)) < 0.18
    for col in ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "serror_rate", "same_srv_rate"]:
        noise = rng.normal(0, df[col].std() * 1.5, len(df))
        df.loc[noisy_rows, col] = (df.loc[noisy_rows, col] + noise[noisy_rows]).clip(lower=0)

    missing_rows = rng.random(len(df)) < 0.01
    df.loc[missing_rows, "dst_bytes"] = np.nan

    # A few mislabeled/ambiguous records happen in real traffic logs too.
    flip_rows = rng.random(len(df)) < 0.02
    df.loc[flip_rows, "label"] = df.loc[flip_rows, "label"].map({"normal": "anomaly", "anomaly": "normal"})

    return df


def clean(df):
    df = df.drop_duplicates()
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    return df


def encode_and_scale(df):
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    scaler = StandardScaler()
    df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])

    # Map label ourselves (normal=0, anomaly=1) instead of LabelEncoder,
    # since LabelEncoder sorts alphabetically and would flip this mapping.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"normal": 0, "anomaly": 1})

    return df, encoders, scaler


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if os.path.exists(RAW_REAL_FILE):
        print(f"Found real NSL-KDD file at {RAW_REAL_FILE} — using it.")
        df = load_real_dataset()
    else:
        print("Real NSL-KDD file not found in data/raw/.")
        print("Using a SYNTHETIC dataset instead (same columns/style as NSL-KDD).")
        print(f"To use real data: download KDDTrain+.txt and place it at {RAW_REAL_FILE}, then rerun this script.")
        df = generate_synthetic_dataset()
        df.to_csv(RAW_SYNTHETIC_FILE, index=False)

    print(f"Loaded {len(df)} rows. Class counts:\n{df['label'].value_counts().to_string()}")

    # Save quick stats for the Streamlit dashboard cards
    total = len(df)
    normal_count = int((df["label"] == "normal").sum())
    anomaly_count = int((df["label"] == "anomaly").sum())
    summary = {
        "total_records": total,
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "attack_percentage": round(anomaly_count / total * 100, 2),
    }
    os.makedirs("reports", exist_ok=True)
    import json
    with open("reports/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    df = clean(df)
    df, encoders, scaler = encode_and_scale(df)

    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    df.to_csv(PROCESSED_FILE, index=False)

    print(f"Processed data saved to {PROCESSED_FILE}")


if __name__ == "__main__":
    main()
