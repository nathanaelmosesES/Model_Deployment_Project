import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from credit_scoring.config import ARTIFACT_DIR  # noqa: E402
from credit_scoring.inference import predict_record  # noqa: E402

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
TESTS_DIR = ROOT / "tests"

# Per-class example payloads (one representative test case per label).
EXAMPLE_FILES = {
    "Good": "sample_payload_good.json",
    "Standard": "sample_payload_standard.json",
    "Poor": "sample_payload_poor.json",
}

NUMERIC_FIELDS = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]
TEXT_FIELDS = [
    "Occupation",
    "Type_of_Loan",
    "Credit_Mix",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]


@st.cache_data
def load_example(label: str) -> dict:
    path = TESTS_DIR / EXAMPLE_FILES[label]
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="Credit Scoring", page_icon="💳", layout="centered")
st.title("💳 Credit Scoring Prediction")
st.caption(
    "Memprediksi kelas `Credit_Score` nasabah (Good / Standard / Poor) "
    "menggunakan model machine learning hasil training MLflow."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model belum ditemukan di `{MODEL_PATH}`.\n\n"
        "Jalankan training dulu: `python scripts/train_local.py --data data_C.csv`"
    )
    st.stop()

# --- Example loader ---------------------------------------------------------
st.subheader("1. Muat contoh per kelas (opsional)")
cols = st.columns(len(EXAMPLE_FILES))
for col, label in zip(cols, EXAMPLE_FILES):
    if col.button(f"Contoh: {label}", use_container_width=True):
        st.session_state["payload"] = load_example(label)

payload = st.session_state.get("payload", load_example("Standard"))

# --- Input form -------------------------------------------------------------
st.subheader("2. Profil finansial nasabah")
with st.form("credit_form"):
    values: dict = {}
    grid = st.columns(2)
    for i, field in enumerate(NUMERIC_FIELDS):
        with grid[i % 2]:
            values[field] = st.number_input(
                field, value=float(payload.get(field, 0) or 0), step=1.0
            )
    for i, field in enumerate(TEXT_FIELDS):
        with grid[i % 2]:
            values[field] = st.text_input(
                field, value=str(payload.get(field, "") or "")
            )
    submitted = st.form_submit_button(
        "Prediksi", use_container_width=True, type="primary"
    )

# --- Prediction -------------------------------------------------------------
if submitted:
    record = {k: (v if v != "" else None) for k, v in values.items()}
    try:
        result = predict_record(record, model_path=MODEL_PATH)
    except Exception as exc:  # surface errors in the UI instead of crashing
        st.exception(exc)
        st.stop()

    prediction = result["prediction"]
    color = {"Good": "green", "Standard": "orange", "Poor": "red"}.get(
        prediction, "blue"
    )
    st.subheader("3. Hasil")
    st.markdown(f"### Prediksi: :{color}[{prediction}]")

    probabilities = result.get("probabilities")
    if probabilities:
        prob_df = pd.DataFrame(
            sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True),
            columns=["class", "probability"],
        ).set_index("class")
        st.bar_chart(prob_df)
        st.dataframe(prob_df, use_container_width=True)

    with st.expander("Lihat payload yang dikirim"):
        st.json(record)
