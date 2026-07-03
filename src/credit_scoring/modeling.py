
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


class _XGBStringLabelClassifier:
    """Wrapper di atas ``XGBClassifier`` agar kompatibel dengan label kelas bertipe string.

    XGBoost versi 3.x ke atas menolak label non-integer saat training. Kelas ini
    menyelipkan ``LabelEncoder`` secara transparan: label string di-encode ke integer
    sebelum dikirim ke XGBoost, lalu di-decode kembali ke string saat prediksi,
    sehingga sisa pipeline tetap bekerja dengan label ``"Good"``, ``"Standard"``,
    ``"Poor"`` secara langsung.
    """

    def __init__(self, **kwargs):
        """Inisialisasi estimator XGBoost internal dengan hyperparameter yang diberikan.

        Parameters
        ----------
        **kwargs:
            Semua argumen diteruskan langsung ke ``XGBClassifier``.
            Contoh: ``n_estimators``, ``learning_rate``, ``max_depth``, dll.
        """
        self._params = kwargs
        self._estimator = XGBClassifier(**kwargs)
        self._le = None
        self.classes_ = None

    def get_params(self, deep: bool = True) -> dict:
        """Kembalikan dict hyperparameter untuk kompatibilitas API sklearn (misal GridSearchCV).

        Parameters
        ----------
        deep:
            Diabaikan — disertakan agar signature sesuai konvensi sklearn.

        Returns
        -------
        dict
            Salinan dict parameter yang dipakai saat inisialisasi.
        """
        return self._params.copy()

    def set_params(self, **params):
        """Perbarui hyperparameter estimator internal secara langsung.

        Dibutuhkan agar sklearn ``Pipeline`` bisa meneruskan parameter saat
        ``set_params`` dipanggil dari luar (misal saat hyperparameter tuning).

        Parameters
        ----------
        **params:
            Pasangan nama-nilai hyperparameter yang ingin diubah.

        Returns
        -------
        self
        """
        self._params.update(params)
        self._estimator.set_params(**params)
        return self

    def fit(self, X, y, **kwargs):
        """Encode label string ke integer lalu latih ``XGBClassifier``.

        Alur:
        1. ``LabelEncoder`` di-fit pada ``y`` untuk membuat mapping string → integer.
        2. ``y`` ditransformasi ke integer dan dikirim ke ``XGBClassifier.fit()``.
        3. ``classes_`` disimpan sebagai array label string asli agar ``predict_proba``
           bisa dipasangkan dengan nama kelas yang benar.

        Parameters
        ----------
        X:
            Matriks fitur yang sudah diproses oleh preprocessor.
        y:
            Label string (``"Good"``, ``"Standard"``, ``"Poor"``).
        **kwargs:
            Argumen tambahan diteruskan ke ``XGBClassifier.fit()``.

        Returns
        -------
        self
        """
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
        y_int = self._le.fit_transform(y)
        self._estimator.fit(X, y_int, **kwargs)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        """Prediksi kelas dan decode hasil integer kembali ke label string.

        Parameters
        ----------
        X:
            Matriks fitur data baru.

        Returns
        -------
        np.ndarray
            Array label string hasil prediksi (``"Good"``, ``"Standard"``, ``"Poor"``).
        """
        y_int = self._estimator.predict(X)
        return self._le.inverse_transform(y_int.astype(int))

    def predict_proba(self, X):
        """Kembalikan probabilitas per kelas langsung dari ``XGBClassifier``.

        Urutan kolom probabilitas sesuai dengan ``self.classes_`` (urutan
        alphabetical setelah LabelEncoder di-fit).

        Parameters
        ----------
        X:
            Matriks fitur data baru.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Probabilitas prediksi untuk setiap kelas.
        """
        return self._estimator.predict_proba(X)


def candidate_models(random_state: int) -> dict:
    models: dict = {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        # L1 logistic regression (lrl1 in FLAML) — fast linear baseline
        "logistic_regression_l1": LogisticRegression(
            penalty="l1",
            C=0.1,
            solver="saga",
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    if _HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.08,
            num_leaves=31,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )

    if _HAS_XGB:
        models["xgboost"] = _XGBStringLabelClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            n_jobs=-1,
            random_state=random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )

    return models


class ModelTrainer:
    """Membungkus satu estimator kandidat beserta preprocessor-nya menjadi sklearn ``Pipeline``.

    Setiap instance mewakili satu model kandidat (misal ExtraTrees, RandomForest, dst.)
    dalam eksperimen. Kelas ini menyediakan antarmuka seragam untuk cross-validation,
    training penuh, dan prediksi, sehingga ``ExperimentRunner`` bisa melatih semua
    kandidat dengan cara yang sama tanpa peduli jenis estimatornya.
    """

    def __init__(self, name: str, estimator, preprocessor: ColumnTransformer):
        """Inisialisasi pipeline dari preprocessor + estimator dan siapkan tempat menyimpan skor CV.

        Parameters
        ----------
        name:
            Nama unik model ini (misal ``"extra_trees"``), dipakai untuk logging MLflow.
        estimator:
            Instance classifier sklearn (atau kompatibel sklearn) yang belum dilatih.
        preprocessor:
            ``ColumnTransformer`` yang sudah dikonfigurasi dengan pipeline
            numerik dan kategorik dari ``CreditPreprocessor.build_transformer()``.
        """
        self.name = name
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        self.cv_scores: np.ndarray | None = None

    @property
    def params(self) -> dict:
        """Kembalikan hyperparameter estimator sebagai dict untuk logging MLflow.

        Returns
        -------
        dict
            Dict hyperparameter dari estimator (hasil ``get_params()``).
        """
        return self.estimator.get_params()

    def cross_validate(self, X, y, cv, scoring: str = "f1_macro") -> np.ndarray:
        """Jalankan cross-validation pada pipeline dan simpan skor ke ``self.cv_scores``.

        Menggunakan ``cross_val_score`` sklearn dengan parallelisasi penuh (``n_jobs=-1``).
        Skor default adalah ``f1_macro`` karena dataset memiliki tiga kelas yang tidak seimbang.

        Parameters
        ----------
        X:
            Matriks fitur data training (belum diproses preprocessor).
        y:
            Label training bertipe string.
        cv:
            Objek CV splitter, biasanya ``StratifiedKFold``.
        scoring:
            Metrik yang digunakan untuk menilai tiap fold. Default ``"f1_macro"``.

        Returns
        -------
        np.ndarray
            Array skor per fold. Gunakan ``.mean()`` dan ``.std()`` untuk ringkasan.
        """
        self.cv_scores = cross_val_score(
            self.pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        return self.cv_scores

    def fit(self, X, y) -> Pipeline:
        """Latih pipeline penuh (preprocessor + estimator) pada seluruh data training.

        Dipanggil setelah cross-validation selesai untuk menghasilkan model final
        yang akan dievaluasi pada test set dan disimpan ke disk.

        Parameters
        ----------
        X:
            Matriks fitur data training.
        y:
            Label training bertipe string.

        Returns
        -------
        Pipeline
            Pipeline sklearn yang sudah dilatih, siap untuk ``predict()``.
        """
        self.pipeline.fit(X, y)
        return self.pipeline

    def predict(self, X):
        """Prediksi kelas untuk data baru menggunakan pipeline yang sudah dilatih.

        Preprocessing (imputasi, scaling, OHE) dijalankan otomatis oleh pipeline
        sebelum data masuk ke estimator.

        Parameters
        ----------
        X:
            Matriks fitur data baru (boleh memiliki nilai hilang).

        Returns
        -------
        np.ndarray
            Array label prediksi (``"Good"``, ``"Standard"``, ``"Poor"``).
        """
        return self.pipeline.predict(X)
