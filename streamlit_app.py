import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"

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

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")


@st.cache_data
def load_example(label: str) -> dict:
    path = TESTS_DIR / EXAMPLE_FILES[label]
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="Credit Scoring", page_icon="💳", layout="centered")
st.title("💳 Credit Scoring Prediction")
st.caption(
    "Memprediksi kelas `Credit_Score` nasabah (Good / Standard / Poor) "
    "via FastAPI backend."
)

# --- Backend URL config ---------------------------------------------------------
with st.sidebar:
    st.header("Backend")
    backend_url = st.text_input(
        "FastAPI URL",
        value=DEFAULT_BACKEND,
        help="URL FastAPI yang berjalan di EC2, contoh: http://1.2.3.4:8000",
    )
    if st.button("Cek koneksi", use_container_width=True):
        try:
            r = requests.get(f"{backend_url}/health", timeout=5)
            r.raise_for_status()
            st.success("Backend online ✓")
        except Exception as e:
            st.error(f"Backend tidak terjangkau: {e}")

# --- Example loader -------------------------------------------------------------
st.subheader("1. Muat contoh per kelas (opsional)")
cols = st.columns(len(EXAMPLE_FILES))
for col, label in zip(cols, EXAMPLE_FILES):
    if col.button(f"Contoh: {label}", use_container_width=True):
        st.session_state["payload"] = load_example(label)

payload = st.session_state.get("payload", load_example("Standard"))

# --- Input form -----------------------------------------------------------------
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

# --- Prediction via FastAPI -----------------------------------------------------
if submitted:
    record = {k: (v if v != "" else None) for k, v in values.items()}
    try:
        response = requests.post(
            f"{backend_url}/predict",
            json=record,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Tidak bisa konek ke backend `{backend_url}`. Pastikan FastAPI sudah jalan.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"Backend error {response.status_code}: {response.text}")
        st.stop()
    except Exception as e:
        st.exception(e)
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
