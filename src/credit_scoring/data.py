import pandas as pd

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from credit_scoring.features import build_features


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def prepare_dataset(path: str):
    raw = load_dataset(path)
    df = build_features(raw)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target `{TARGET_COLUMN}` tidak ditemukan.")

    y = df[TARGET_COLUMN].astype("string").str.strip()
    available_numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    available_categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    features = available_numeric + available_categorical

    if not features:
        raise ValueError("Tidak ada fitur yang cocok dengan schema pipeline.")

    X = df[features].copy()
    metadata = {
        "target": TARGET_COLUMN,
        "dropped_columns": [col for col in ID_COLUMNS if col in df.columns],
        "numeric_features": available_numeric,
        "categorical_features": available_categorical,
        "features": features,
        "n_rows": int(len(df)),
    }
    return X, y, metadata

