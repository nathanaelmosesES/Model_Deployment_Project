"""Lapisan evaluasi model.

``MetricSet`` menyimpan lima metrik scalar; ``Evaluator`` adalah kelas OOP utama
yang menghasilkan metrik scalar maupun laporan lengkap (classification report +
confusion matrix) untuk model terbaik.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class MetricSet:
    """Container tidak-mutable untuk lima metrik klasifikasi scalar hasil evaluasi model.

    Disimpan sebagai dataclass agar mudah dikonversi ke dict (untuk MLflow)
    maupun diakses per field secara eksplisit.

    Attributes
    ----------
    accuracy:
        Proporsi prediksi yang benar dari total sampel.
    precision_macro:
        Rata-rata presisi per kelas tanpa pembobotan jumlah sampel.
    recall_macro:
        Rata-rata recall per kelas tanpa pembobotan jumlah sampel.
    f1_macro:
        Rata-rata F1-score per kelas tanpa pembobotan — metrik utama seleksi model
        karena menghukum model yang mengabaikan kelas minoritas.
    f1_weighted:
        Rata-rata F1-score per kelas dibobot proporsional jumlah sampel per kelas.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float

    def to_dict(self) -> dict:
        """Konversi seluruh field ke dict Python biasa.

        Returns
        -------
        dict
            Dict berisi semua metrik dengan nama field sebagai key.
            Cocok untuk diteruskan ke ``mlflow.log_metrics()``.
        """
        return asdict(self)


class Evaluator:
    """Menghitung dan menyusun metrik klasifikasi dari hasil prediksi model.

    Menyediakan dua tingkat output:
    - **Scalar** (via ``metrics``): dict ringkas untuk logging MLflow dan seleksi model.
    - **Laporan lengkap** (via ``report``): termasuk classification report per kelas
      dan confusion matrix, dipakai untuk model terbaik saja.
    """

    def metric_set(self, y_true, y_pred) -> MetricSet:
        """Hitung lima metrik klasifikasi dan kembalikan sebagai ``MetricSet``.

        Parameters
        ----------
        y_true:
            Label aktual dari test set.
        y_pred:
            Label hasil prediksi model.

        Returns
        -------
        MetricSet
            Object berisi accuracy, precision_macro, recall_macro,
            f1_macro, dan f1_weighted.
        """
        return MetricSet(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision_macro=float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            recall_macro=float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            f1_weighted=float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        )

    def metrics(self, y_true, y_pred) -> dict:
        """Kembalikan metrik scalar sebagai dict biasa, siap untuk ``mlflow.log_metrics()``.

        Parameters
        ----------
        y_true:
            Label aktual dari test set.
        y_pred:
            Label hasil prediksi model.

        Returns
        -------
        dict
            Dict metrik (accuracy, precision_macro, recall_macro, f1_macro, f1_weighted).
        """
        return self.metric_set(y_true, y_pred).to_dict()

    def report(self, y_true, y_pred) -> dict:
        """Buat laporan evaluasi lengkap untuk model terbaik yang akan disimpan ke ``metrics.json``.

        Berisi lebih dari sekadar metrik scalar: menambahkan classification report
        per kelas (precision/recall/f1 per ``"Good"``, ``"Standard"``, ``"Poor"``)
        dan confusion matrix dalam format list 2D.

        Parameters
        ----------
        y_true:
            Label aktual dari test set.
        y_pred:
            Label hasil prediksi model terbaik.

        Returns
        -------
        dict
            Dict dengan key:
            - ``"test_metrics"``: dict metrik scalar.
            - ``"classification_report"``: dict per-kelas dari sklearn.
            - ``"confusion_matrix"``: list 2D (baris = aktual, kolom = prediksi).
            - ``"labels"``: daftar nama kelas yang diurutkan alphabetically.
        """
        labels = sorted(set(y_true))
        return {
            "test_metrics": self.metrics(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "labels": labels,
        }


# Backward-compatible function aliases (kept so existing imports keep working).
def classification_metrics(y_true, y_pred) -> MetricSet:
    return Evaluator().metric_set(y_true, y_pred)


def metrics_to_dict(metrics: MetricSet) -> dict:
    return metrics.to_dict()
