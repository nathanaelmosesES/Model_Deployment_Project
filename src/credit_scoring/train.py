from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from credit_scoring.config import ARTIFACT_DIR, PROJECT_ROOT
from credit_scoring.evaluate import Evaluator
from credit_scoring.modeling import ModelTrainer, candidate_models
from credit_scoring.preprocessing import CreditPreprocessor

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is not installed.
    tqdm = None


DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT = "credit_scoring"


class SimpleProgress:
    def __init__(self, total: int, enabled: bool = True):
        self.total = total
        self.enabled = enabled
        self.current = 0

    def __enter__(self):
        if self.enabled:
            print(f"Training progress: 0/{self.total} (0.0%)", file=sys.stderr)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.enabled and exc_type is None and self.current < self.total:
            self.update(self.total - self.current, stage="done")

    def set_description(self, description: str):
        if self.enabled:
            print(description, file=sys.stderr)

    def set_postfix(self, **kwargs):
        if not self.enabled or not kwargs:
            return
        details = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        print(f"  {details}", file=sys.stderr)

    def update(self, n: int = 1, **kwargs):
        self.current += n
        if not self.enabled:
            return
        percent = (self.current / self.total) * 100
        details = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        suffix = f" | {details}" if details else ""
        print(
            f"Training progress: {self.current}/{self.total} ({percent:.1f}%){suffix}",
            file=sys.stderr,
        )


def make_progress(total: int, enabled: bool):
    if tqdm is None:
        return SimpleProgress(total=total, enabled=enabled)
    return tqdm(
        total=total,
        disable=not enabled,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )


class ExperimentRunner:
    def __init__(
        self,
        data_path: str,
        output_dir: str | Path = ARTIFACT_DIR,
        random_state: int = 42,
        test_size: float = 0.2,
        cv_splits: int = 3,
        show_progress: bool = True,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        experiment_name: str = DEFAULT_EXPERIMENT,
    ):
        self.data_path = data_path
        self.output = Path(output_dir)
        self.random_state = random_state
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.show_progress = show_progress
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        self.preprocessor = CreditPreprocessor()
        self.evaluator = Evaluator()

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        raw = pd.read_csv(self.data_path, low_memory=False)
        X, y, metadata = self.preprocessor.prepare(raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        cv = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )
        models = candidate_models(self.random_state)
        total_steps = 1 + len(models) * 2 + 3

        rows = []
        trained: dict[str, ModelTrainer] = {}

        with mlflow.start_run(run_name="training_session") as parent_run:
            mlflow.set_tags(
                {
                    "stage": "training",
                    "n_models": len(models),
                    "data_path": str(self.data_path),
                }
            )
            mlflow.log_params(
                {
                    "random_state": self.random_state,
                    "test_size": self.test_size,
                    "cv_splits": self.cv_splits,
                    "n_rows": metadata["n_rows"],
                    "n_features": len(metadata["features"]),
                }
            )

            with make_progress(
                total=total_steps, enabled=self.show_progress
            ) as progress:
                progress.set_description("Preparing data")
                progress.update(1)

                for name, estimator in models.items():
                    trainer = ModelTrainer(
                        name=name,
                        estimator=estimator,
                        preprocessor=self.preprocessor.build_transformer(),
                    )

                    with mlflow.start_run(run_name=name, nested=True):
                        mlflow.log_param("model_name", name)
                        mlflow.log_params(self._loggable_params(trainer.params))

                        progress.set_description(f"{name}: cross-validation")
                        cv_scores = trainer.cross_validate(X_train, y_train, cv=cv)
                        mlflow.log_metric("cv_f1_macro_mean", float(cv_scores.mean()))
                        mlflow.log_metric("cv_f1_macro_std", float(cv_scores.std()))
                        progress.set_postfix(
                            model=name,
                            cv_f1_macro_mean=f"{cv_scores.mean():.4f}",
                        )
                        progress.update(1)

                        progress.set_description(f"{name}: fit + evaluate")
                        trainer.fit(X_train, y_train)
                        trained[name] = trainer
                        pred = trainer.predict(X_test)
                        metrics = self.evaluator.metrics(y_test, pred)
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(
                            trainer.pipeline,
                            name="model",
                            serialization_format="cloudpickle",
                        )

                        row = {
                            "model_name": name,
                            "cv_f1_macro_mean": float(cv_scores.mean()),
                            "cv_f1_macro_std": float(cv_scores.std()),
                            **metrics,
                        }
                        rows.append(row)
                        progress.set_postfix(
                            model=name,
                            test_f1_macro=f"{row['f1_macro']:.4f}",
                            cv_f1_macro_mean=f"{row['cv_f1_macro_mean']:.4f}",
                        )
                        progress.update(1)

                progress.set_description("Selecting best model")
                results = pd.DataFrame(rows).sort_values(
                    ["f1_macro", "cv_f1_macro_mean"], ascending=False
                )
                best_name = str(results.iloc[0]["model_name"])
                best_trainer = trained[best_name]
                best_pred = best_trainer.predict(X_test)
                progress.set_postfix(best_model=best_name)
                progress.update(1)

                progress.set_description("Building metrics")
                metadata.update(
                    {
                        "best_model": best_name,
                        "random_state": self.random_state,
                        "test_size": self.test_size,
                        "cv_splits": self.cv_splits,
                        "classes": sorted(y.dropna().unique().tolist()),
                    }
                )
                report = self.evaluator.report(y_test, best_pred)
                metrics = {"best_model": best_name, **report}
                progress.update(1)

                progress.set_description("Saving artifacts")
                self._save_artifacts(best_trainer.pipeline, results, metrics, metadata)
                progress.update(1)

            # Promote the winning model to the parent run.
            mlflow.set_tag("best_model", best_name)
            mlflow.log_metric("best_f1_macro", float(results.iloc[0]["f1_macro"]))
            mlflow.sklearn.log_model(
                best_trainer.pipeline,
                name="best_model",
                serialization_format="cloudpickle",
            )
            mlflow.log_artifact(str(self.output / "metrics.json"))
            mlflow.log_artifact(str(self.output / "experiment_results.csv"))
            run_id = parent_run.info.run_id

        return {
            "best_model": best_name,
            "metrics": metrics,
            "experiment_results": results.to_dict(orient="records"),
            "artifact_dir": str(self.output.resolve()),
            "mlflow_run_id": run_id,
            "mlflow_tracking_uri": self.tracking_uri,
        }

    @staticmethod
    def _loggable_params(params: dict) -> dict:
        """Keep only scalar hyperparameters MLflow can store cleanly."""
        return {
            key: value
            for key, value in params.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    def _save_artifacts(
        self, model, results: pd.DataFrame, metrics: dict, metadata: dict
    ):
        joblib.dump(model, self.output / "model.joblib")
        results.to_csv(self.output / "experiment_results.csv", index=False)
        (self.output / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (self.output / "feature_schema.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        run_log = {
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_path": str(self.data_path),
            "artifact_dir": str(self.output.resolve()),
            "best_model": metadata["best_model"],
            "best_f1_macro": float(results.iloc[0]["f1_macro"]),
            "best_cv_f1_macro_mean": float(results.iloc[0]["cv_f1_macro_mean"]),
            "n_rows": metadata["n_rows"],
            "features": metadata["features"],
        }
        with (self.output / "training_runs.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(run_log) + "\n")


def train_experiments(
    data_path: str,
    output_dir: str | Path = ARTIFACT_DIR,
    random_state: int = 42,
    test_size: float = 0.2,
    show_progress: bool = True,
    cv_splits: int = 3,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT,
):
    """Functional entrypoint kept for backward compatibility with scripts."""
    runner = ExperimentRunner(
        data_path=data_path,
        output_dir=output_dir,
        random_state=random_state,
        test_size=test_size,
        cv_splits=cv_splits,
        show_progress=show_progress,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    return runner.run()
