# Shared column names used by preprocess.py, train.py and predict.py.
# Kept in one place so all three files always agree on the feature list.

NUMERIC_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "serror_rate",
    "same_srv_rate",
]

CATEGORICAL_FEATURES = [
    "protocol_type",
    "service",
    "flag",
]

TARGET_COLUMN = "label"
