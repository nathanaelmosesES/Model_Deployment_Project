
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from credit_scoring.features import build_features


class CreditPreprocessor:
    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        target_column: str = TARGET_COLUMN,
    ):
        self.numeric_features = numeric_features or list(NUMERIC_FEATURES)
        self.categorical_features = categorical_features or list(CATEGORICAL_FEATURES)
        self.target_column = target_column
        self.metadata: dict | None = None

    def engineer(self, raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw)

    def prepare(self, raw: pd.DataFrame):
        df = self.engineer(raw)

        if self.target_column not in df.columns:
            raise ValueError(f"Kolom target `{self.target_column}` tidak ditemukan.")

        y = df[self.target_column].astype("string").str.strip()
        available_numeric = [c for c in self.numeric_features if c in df.columns]
        available_categorical = [c for c in self.categorical_features if c in df.columns]
        features = available_numeric + available_categorical

        if not features:
            raise ValueError("Tidak ada fitur yang cocok dengan schema pipeline.")

        self.numeric_features = available_numeric
        self.categorical_features = available_categorical

        X = df[features].copy()
        self.metadata = {
            "target": self.target_column,
            "dropped_columns": [c for c in ID_COLUMNS if c in df.columns],
            "numeric_features": available_numeric,
            "categorical_features": available_categorical,
            "features": features,
            "n_rows": int(len(df)),
        }
        return X, y, self.metadata

    def build_transformer(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )
