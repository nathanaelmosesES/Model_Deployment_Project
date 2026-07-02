
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
    """Wraps XGBClassifier to handle string class labels (XGBoost 3.x requires integers)."""

    def __init__(self, **kwargs):
        self._params = kwargs
        self._estimator = XGBClassifier(**kwargs)
        self._le = None
        self.classes_ = None

    def get_params(self, deep: bool = True) -> dict:
        return self._params.copy()

    def set_params(self, **params):
        self._params.update(params)
        self._estimator.set_params(**params)
        return self

    def fit(self, X, y, **kwargs):
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
        y_int = self._le.fit_transform(y)
        self._estimator.fit(X, y_int, **kwargs)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        y_int = self._estimator.predict(X)
        return self._le.inverse_transform(y_int.astype(int))

    def predict_proba(self, X):
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

    def __init__(self, name: str, estimator, preprocessor: ColumnTransformer):
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
        return self.estimator.get_params()

    def cross_validate(self, X, y, cv, scoring: str = "f1_macro") -> np.ndarray:
        self.cv_scores = cross_val_score(
            self.pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        return self.cv_scores

    def fit(self, X, y) -> Pipeline:
        self.pipeline.fit(X, y)
        return self.pipeline

    def predict(self, X):
        return self.pipeline.predict(X)
