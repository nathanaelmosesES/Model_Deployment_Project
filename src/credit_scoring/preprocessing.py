
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
    """Mengorkestrasi seluruh tahap pra-pemrosesan data sebelum masuk ke model.

    Tanggung jawab utama:
    - Menjalankan feature engineering pada data mentah (via ``build_features``).
    - Memisahkan fitur (X) dan label (y) sesuai schema yang dikonfigurasi.
    - Membangun ``ColumnTransformer`` sklearn yang berisi pipeline numerik
      (imputasi median + standarisasi) dan pipeline kategorik (imputasi modus + OHE).
    - Menyimpan metadata fitur (daftar kolom, jumlah baris, dll.) untuk diserialisasi
      bersama model ke ``feature_schema.json``.
    """

    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        target_column: str = TARGET_COLUMN,
    ):
        """Inisialisasi preprocessor dengan daftar fitur dan nama kolom target.

        Parameters
        ----------
        numeric_features:
            Daftar nama kolom numerik yang akan diproses. Jika ``None``,
            menggunakan ``NUMERIC_FEATURES`` dari config.
        categorical_features:
            Daftar nama kolom kategorik yang akan diproses. Jika ``None``,
            menggunakan ``CATEGORICAL_FEATURES`` dari config.
        target_column:
            Nama kolom label/target di dataset. Default: ``"Credit_Score"``.
        """
        self.numeric_features = numeric_features or list(NUMERIC_FEATURES)
        self.categorical_features = categorical_features or list(CATEGORICAL_FEATURES)
        self.target_column = target_column
        self.metadata: dict | None = None

    def engineer(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Terapkan seluruh pipeline feature engineering ke dataframe mentah.

        Memanggil ``build_features`` yang menangani: pembersihan nama kolom,
        normalisasi nilai hilang, parsing angka noisy, konversi Credit_History_Age
        ke bulan, clipping outlier domain, pembuatan flag pinjaman, dan fitur rasio.

        Parameters
        ----------
        raw:
            DataFrame mentah hasil baca CSV sebelum diproses sama sekali.

        Returns
        -------
        pd.DataFrame
            DataFrame yang sudah bersih dan diperkaya dengan fitur-fitur baru.
        """
        return build_features(raw)

    def prepare(self, raw: pd.DataFrame):
        """Siapkan X, y, dan metadata dari dataframe mentah untuk proses training.

        Menjalankan feature engineering, lalu memfilter kolom yang benar-benar
        tersedia di data (intersection antara schema dan kolom aktual), sehingga
        pipeline tidak error jika dataset tidak memiliki semua kolom opsional.

        Parameters
        ----------
        raw:
            DataFrame mentah hasil baca CSV.

        Returns
        -------
        X : pd.DataFrame
            Matriks fitur yang siap masuk ``ColumnTransformer``.
        y : pd.Series
            Label target bertipe string (``"Good"``, ``"Standard"``, ``"Poor"``).
        metadata : dict
            Informasi schema: daftar fitur numerik/kategorik, jumlah baris,
            kolom yang di-drop, dsb. Disimpan ke ``feature_schema.json``.

        Raises
        ------
        ValueError
            Jika kolom target tidak ditemukan atau tidak ada fitur yang cocok schema.
        """
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
        """Bangun ``ColumnTransformer`` sklearn dengan dua sub-pipeline terpisah.

        Pipeline numerik:
            1. ``SimpleImputer(strategy="median")`` — isi nilai hilang dengan median kolom.
            2. ``StandardScaler()`` — standarisasi ke mean=0, std=1.

        Pipeline kategorik:
            1. ``SimpleImputer(strategy="most_frequent")`` — isi nilai hilang dengan modus.
            2. ``OneHotEncoder(handle_unknown="ignore")`` — encode ke vektor biner;
               kategori yang tidak dikenal saat inferensi diabaikan (semua nol).

        Kolom di luar daftar numerik/kategorik akan di-drop (``remainder="drop"``).

        Returns
        -------
        ColumnTransformer
            Transformer yang siap dimasukkan sebagai langkah pertama sklearn ``Pipeline``.
        """
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
