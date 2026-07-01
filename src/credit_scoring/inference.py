import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_scoring.config import ARTIFACT_DIR
from credit_scoring.features import build_features


def load_model(model_path: str | Path = ARTIFACT_DIR / "model.joblib"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model belum ditemukan di {path}. Jalankan training terlebih dahulu."
        )
    return joblib.load(path)


def load_schema(schema_path: str | Path = ARTIFACT_DIR / "feature_schema.json") -> dict:
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Schema belum ditemukan di {path}. Jalankan training terlebih dahulu."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def predict_record(payload: dict, model_path: str | Path = ARTIFACT_DIR / "model.joblib"):
    model = load_model(model_path)
    schema = load_schema(Path(model_path).with_name("feature_schema.json"))
    df = pd.DataFrame([payload])
    features = build_features(df)

    for col in schema["features"]:
        if col not in features.columns:
            features[col] = np.nan

    features = features[schema["features"]]
    features = features.replace({pd.NA: np.nan})
    prediction = model.predict(features)[0]
    result = {"prediction": str(prediction)}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        result["probabilities"] = {
            str(label): float(prob)
            for label, prob in zip(model.classes_.tolist(), probabilities)
        }

    return result
