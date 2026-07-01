import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from credit_scoring.inference import predict_record


GOOD_EXAMPLE = {
    "Age": 46,
    "Occupation": "Manager",
    "Annual_Income": 115000,
    "Monthly_Inhand_Salary": 7600,
    "Num_Bank_Accounts": 2,
    "Num_Credit_Card": 3,
    "Interest_Rate": 5,
    "Num_of_Loan": 1,
    "Type_of_Loan": "Home Equity Loan",
    "Delay_from_due_date": 2,
    "Num_of_Delayed_Payment": 0,
    "Changed_Credit_Limit": 2,
    "Num_Credit_Inquiries": 1,
    "Credit_Mix": "Good",
    "Outstanding_Debt": 350,
    "Credit_Utilization_Ratio": 18,
    "Credit_History_Age": "18 Years and 9 Months",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 90,
    "Amount_invested_monthly": 1200,
    "Payment_Behaviour": "High_spent_Large_value_payments",
    "Monthly_Balance": 1800,
}

POOR_EXAMPLE = {
    "Age": 24,
    "Occupation": "Scientist",
    "Annual_Income": 18000,
    "Monthly_Inhand_Salary": 1200,
    "Num_Bank_Accounts": 9,
    "Num_Credit_Card": 8,
    "Interest_Rate": 28,
    "Num_of_Loan": 7,
    "Type_of_Loan": "Personal Loan, Payday Loan",
    "Delay_from_due_date": 45,
    "Num_of_Delayed_Payment": 16,
    "Changed_Credit_Limit": 25,
    "Num_Credit_Inquiries": 12,
    "Credit_Mix": "Bad",
    "Outstanding_Debt": 4200,
    "Credit_Utilization_Ratio": 55,
    "Credit_History_Age": "1 Years and 2 Months",
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 420,
    "Amount_invested_monthly": 20,
    "Payment_Behaviour": "Low_spent_Small_value_payments",
    "Monthly_Balance": 80,
}

STANDARD_EXAMPLE = {
    "Age": 35,
    "Occupation": "Engineer",
    "Annual_Income": 55000,
    "Monthly_Inhand_Salary": 3800,
    "Num_Bank_Accounts": 4,
    "Num_Credit_Card": 4,
    "Interest_Rate": 14,
    "Num_of_Loan": 3,
    "Type_of_Loan": "Auto Loan, Credit-Builder Loan",
    "Delay_from_due_date": 12,
    "Num_of_Delayed_Payment": 4,
    "Changed_Credit_Limit": 8,
    "Num_Credit_Inquiries": 4,
    "Credit_Mix": "Standard",
    "Outstanding_Debt": 1200,
    "Credit_Utilization_Ratio": 32,
    "Credit_History_Age": "8 Years and 5 Months",
    "Payment_of_Min_Amount": "NM",
    "Total_EMI_per_month": 180,
    "Amount_invested_monthly": 250,
    "Payment_Behaviour": "High_spent_Medium_value_payments",
    "Monthly_Balance": 520,
}


class CreditPayload(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [GOOD_EXAMPLE, POOR_EXAMPLE, STANDARD_EXAMPLE],
        },
    )

    Age: int | float | None = Field(None, examples=[35], description="Usia nasabah.")
    Occupation: str | None = Field(None, examples=["Engineer"])
    Annual_Income: float | None = Field(None, examples=[55000])
    Monthly_Inhand_Salary: float | None = Field(None, examples=[3800])
    Num_Bank_Accounts: int | float | None = Field(None, examples=[4])
    Num_Credit_Card: int | float | None = Field(None, examples=[4])
    Interest_Rate: float | None = Field(None, examples=[14])
    Num_of_Loan: int | float | None = Field(None, examples=[3])
    Type_of_Loan: str | None = Field(
        None,
        examples=["Auto Loan, Credit-Builder Loan"],
        description="Daftar tipe pinjaman, dipisahkan koma jika lebih dari satu.",
    )
    Delay_from_due_date: int | float | None = Field(None, examples=[12])
    Num_of_Delayed_Payment: int | float | None = Field(None, examples=[4])
    Changed_Credit_Limit: float | None = Field(None, examples=[8])
    Num_Credit_Inquiries: int | float | None = Field(None, examples=[4])
    Credit_Mix: Literal["Bad", "Standard", "Good"] | None = Field(
        None,
        examples=["Standard"],
    )
    Outstanding_Debt: float | None = Field(None, examples=[1200])
    Credit_Utilization_Ratio: float | None = Field(None, examples=[32])
    Credit_History_Age: str | None = Field(
        None,
        examples=["8 Years and 5 Months"],
        description="Format yang didukung: '<tahun> Years and <bulan> Months'.",
    )
    Payment_of_Min_Amount: Literal["Yes", "No", "NM"] | None = Field(
        None,
        examples=["NM"],
    )
    Total_EMI_per_month: float | None = Field(None, examples=[180])
    Amount_invested_monthly: float | None = Field(None, examples=[250])
    Payment_Behaviour: str | None = Field(
        None,
        examples=["High_spent_Medium_value_payments"],
    )
    Monthly_Balance: float | None = Field(None, examples=[520])
    Month: str | None = Field(
        None,
        examples=["June"],
        description="Opsional. Nama bulan transaksi jika tersedia.",
    )


class PredictionResponse(BaseModel):
    prediction: Literal["Good", "Poor", "Standard"] = Field(
        examples=["Standard"],
        description="Kelas credit score hasil prediksi model.",
    )
    probabilities: dict[str, float] | None = Field(
        None,
        examples=[{"Good": 0.16, "Poor": 0.16, "Standard": 0.68}],
        description="Probabilitas tiap kelas jika model mendukung predict_proba.",
    )


app = FastAPI(
    title="Credit Scoring Model API",
    version="1.1.0",
    description=(
        "API lokal untuk memprediksi kelas `Credit_Score` nasabah: "
        "`Good`, `Poor`, atau `Standard`. Buka endpoint `/predict`, klik "
        "`Try it out`, lalu gunakan salah satu contoh input yang tersedia."
    ),
    contact={"name": "Credit Scoring Project"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    summary="Cek status API",
    tags=["System"],
)
def health():
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prediksi credit score",
    description=(
        "Kirim data profil finansial nasabah. Field yang tidak tersedia boleh "
        "dihilangkan; pipeline akan melakukan imputasi nilai kosong."
    ),
    tags=["Prediction"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "good_customer": {
                            "summary": "Contoh nasabah berisiko rendah",
                            "value": GOOD_EXAMPLE,
                        },
                        "poor_customer": {
                            "summary": "Contoh nasabah berisiko tinggi",
                            "value": POOR_EXAMPLE,
                        },
                        "standard_customer": {
                            "summary": "Contoh nasabah standar",
                            "value": STANDARD_EXAMPLE,
                        },
                    }
                }
            }
        }
    },
)
def predict(payload: CreditPayload):
    return predict_record(payload.model_dump())
