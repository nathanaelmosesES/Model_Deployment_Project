"""Evaluation layer.

`MetricSet` holds the scalar metrics; `Evaluator` is the OOP "evaluation" class
required by the brief, producing both the scalar metric set and the richer
report (classification report + confusion matrix) used for the best model.
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
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    """Computes classification metrics for a model's predictions."""

    def metric_set(self, y_true, y_pred) -> MetricSet:
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
        """Scalar metrics as a plain dict (handy for MLflow log_metrics)."""
        return self.metric_set(y_true, y_pred).to_dict()

    def report(self, y_true, y_pred) -> dict:
        """Full evaluation report for the best model."""
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
