# Credit Scoring — ML Pipeline

Classifies a customer's credit score into **Good**, **Standard**, or **Poor** using tabular financial features. Exposes predictions via a FastAPI REST API.

---

## Pipeline Overview

```
data_C.csv
    │
    ▼
[1] Feature Engineering          features.py
    ├── clean column names & normalize missing values
    ├── parse noisy numerics (e.g. "1,000_" → 1000.0)
    ├── parse Credit_History_Age → months (integer)
    ├── clip domain outliers (e.g. Age capped at 18–100)
    ├── expand Type_of_Loan → 8 binary Has_*_Loan flags + Loan_Type_Count
    └── compute ratio features: Debt_To_Income, EMI_To_Salary,
        Investment_To_Salary, Balance_To_Salary
    │
    ▼
[2] Preprocessing                preprocessing.py  →  CreditPreprocessor
    ├── Numeric (30 features)
    │   ├── SimpleImputer (median)
    │   └── StandardScaler
    └── Categorical (5 features: Occupation, Credit_Mix,
        Payment_of_Min_Amount, Payment_Behaviour, Month)
        ├── SimpleImputer (most_frequent)
        └── OneHotEncoder (handle_unknown="ignore")
    │
    ▼
[3] Model Selection              modeling.py  →  ModelTrainer + ExperimentRunner
    ├── Candidate models trained in parallel:
    │   ├── ExtraTreesClassifier      (primary — usually wins)
    │   ├── RandomForestClassifier
    │   ├── LogisticRegression (L1)   (linear baseline)
    │   ├── LGBMClassifier            (if lightgbm installed)
    │   └── XGBClassifier             (if xgboost installed, via _XGBStringLabelClassifier)
    ├── StratifiedKFold cross-validation (default 3 folds, scored by f1_macro)
    ├── Final fit on full training set
    └── Best model selected by: f1_macro (test) then cv_f1_macro_mean
    │
    ▼
[4] Evaluation                   evaluate.py  →  Evaluator + MetricSet
    ├── accuracy, precision_macro, recall_macro
    ├── f1_macro, f1_weighted
    ├── full classification report (per-class precision/recall/f1)
    └── confusion matrix
    │
    ▼
[5] Artifact Persistence         train.py  →  ExperimentRunner._save_artifacts
    ├── artifacts/model.joblib          ← sklearn Pipeline (preprocessor + best model)
    ├── artifacts/feature_schema.json   ← feature list, classes, metadata
    ├── artifacts/metrics.json          ← evaluation report of the best model
    ├── artifacts/experiment_results.csv← comparison of all candidates
    └── mlruns/                         ← MLflow experiment tracking
    │
    ▼
[6] Inference                    inference.py
    ├── Load model.joblib + feature_schema.json
    ├── Apply feature engineering to raw input dict
    ├── Align columns to schema (fill missing with NaN)
    └── Return prediction + predict_proba (if supported)
    │
    ▼
[7] REST API                     scripts/serve_api.py
    ├── POST /predict  →  CreditPayload → predict_record() → PredictionResponse
    └── GET  /health   →  {"status": "ok"}
```

---

## Project Structure

```
modelling/
├── src/credit_scoring/
│   ├── config.py          feature lists, paths, constants
│   ├── features.py        raw data cleaning & feature engineering
│   ├── preprocessing.py   sklearn transformer builder (CreditPreprocessor)
│   ├── modeling.py        candidate models & ModelTrainer
│   ├── train.py           ExperimentRunner — full training loop + MLflow
│   ├── evaluate.py        Evaluator, MetricSet
│   └── inference.py       load artifacts & predict a single record
├── scripts/
│   ├── train_local.py     CLI entrypoint for training
│   ├── serve_api.py       FastAPI app
│   └── predict_one.py     CLI for single-record prediction
└── artifacts/             generated at training time — not committed to git
```

---

## Quickstart

### Train

```bash
python scripts/train_local.py --data data_C.csv --output-dir artifacts
```

| Flag | Default | Description |
|---|---|---|
| `--data` | `data_C.csv` | Path to CSV dataset |
| `--output-dir` | `artifacts` | Where to save model & schema |
| `--cv-splits` | `3` | Number of cross-validation folds |
| `--random-state` | `42` | Reproducibility seed |

### Serve

```bash
uvicorn scripts.serve_api:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Predict (CLI)

```bash
python scripts/predict_one.py
```

---

## Key Design Decisions

- **No artifacts in git** — `artifacts/` is fully gitignored. Model and schema are distributed via S3 for cloud deployments (see `docs/AWS_DEPLOY.md`).
- **All fields optional at inference** — `CreditPayload` accepts partial records; the pipeline imputes missing values at predict time using the same strategy used during training.
- **Model selection by f1_macro** — chosen over accuracy because the three credit classes (Good/Standard/Poor) are imbalanced; macro-F1 penalises models that ignore minority classes.
- **`_XGBStringLabelClassifier` wrapper** — XGBoost 3.x dropped support for string labels; this wrapper transparently encodes/decodes labels so the rest of the pipeline stays string-native.
