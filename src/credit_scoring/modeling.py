
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def candidate_models(random_state: int) -> dict:
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            random_state=random_state,
        ),
    }


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
