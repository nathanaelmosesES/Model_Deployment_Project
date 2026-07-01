import re

import numpy as np
import pandas as pd

from credit_scoring.config import RAW_NUMERIC_COLUMNS


DOMAIN_LIMITS = {
    "Age": (18, 100),
    "Annual_Income": (0, 1_000_000),
    "Monthly_Inhand_Salary": (0, 50_000),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 100),
    "Num_of_Loan": (0, 20),
    "Delay_from_due_date": (0, 90),
    "Num_of_Delayed_Payment": (0, 100),
    "Changed_Credit_Limit": (-20, 100),
    "Num_Credit_Inquiries": (0, 100),
    "Outstanding_Debt": (0, 100_000),
    "Credit_Utilization_Ratio": (0, 100),
    "Total_EMI_per_month": (0, 50_000),
    "Amount_invested_monthly": (0, 20_000),
    "Monthly_Balance": (0, 50_000),
}

LOAN_TYPE_FLAGS = {
    "Auto Loan": "Has_Auto_Loan",
    "Credit-Builder Loan": "Has_Credit_Builder_Loan",
    "Debt Consolidation Loan": "Has_Debt_Consolidation_Loan",
    "Home Equity Loan": "Has_Home_Equity_Loan",
    "Mortgage Loan": "Has_Mortgage_Loan",
    "Payday Loan": "Has_Payday_Loan",
    "Personal Loan": "Has_Personal_Loan",
    "Student Loan": "Has_Student_Loan",
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    missing_like = [
        "",
        " ",
        "NA",
        "N/A",
        "nan",
        "None",
        "null",
        "!@9#%8",
    ]
    cleaned = cleaned.replace(missing_like, np.nan)
    cleaned = cleaned.replace(r"^_+$", np.nan, regex=True)
    return cleaned


def parse_numeric(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    text = text.replace("_", "")
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return np.nan
    return float(match.group(0))


def parse_credit_history_age(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    years = re.search(r"(\d+)\s*years?", text)
    months = re.search(r"(\d+)\s*months?", text)
    total_months = 0
    found = False
    if years:
        total_months += int(years.group(1)) * 12
        found = True
    if months:
        total_months += int(months.group(1))
        found = True
    return float(total_months) if found else parse_numeric(value)


def clip_domain_outliers(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col, (lower, upper) in DOMAIN_LIMITS.items():
        if col not in cleaned.columns:
            continue
        cleaned[col] = cleaned[col].where(cleaned[col].between(lower, upper))
    return cleaned


def add_loan_type_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "Type_of_Loan" not in enriched.columns:
        return enriched

    loan_text = enriched["Type_of_Loan"].fillna("").astype(str)
    enriched["Loan_Type_Count"] = loan_text.map(
        lambda value: 0
        if not value
        else len(
            [
                part
                for part in re.split(r",| and ", value)
                if part.strip() and part.strip() != "Not Specified"
            ]
        )
    )

    for loan_name, feature_name in LOAN_TYPE_FLAGS.items():
        enriched[feature_name] = loan_text.str.contains(
            re.escape(loan_name),
            case=False,
            regex=True,
        ).astype(int)

    return enriched


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if {"Outstanding_Debt", "Annual_Income"}.issubset(enriched.columns):
        enriched["Debt_To_Income"] = safe_ratio(
            enriched["Outstanding_Debt"],
            enriched["Annual_Income"],
        )
    if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(enriched.columns):
        enriched["EMI_To_Salary"] = safe_ratio(
            enriched["Total_EMI_per_month"],
            enriched["Monthly_Inhand_Salary"],
        )
    if {"Amount_invested_monthly", "Monthly_Inhand_Salary"}.issubset(
        enriched.columns
    ):
        enriched["Investment_To_Salary"] = safe_ratio(
            enriched["Amount_invested_monthly"],
            enriched["Monthly_Inhand_Salary"],
        )
    if {"Monthly_Balance", "Monthly_Inhand_Salary"}.issubset(enriched.columns):
        enriched["Balance_To_Salary"] = safe_ratio(
            enriched["Monthly_Balance"],
            enriched["Monthly_Inhand_Salary"],
        )
    return enriched


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_column_names(df)
    cleaned = normalize_missing_values(cleaned)

    for col in RAW_NUMERIC_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].map(parse_numeric)

    if "Credit_History_Age" in cleaned.columns:
        cleaned["Credit_History_Age_Months"] = cleaned["Credit_History_Age"].map(
            parse_credit_history_age
        )

    cleaned = clip_domain_outliers(cleaned)
    cleaned = add_loan_type_features(cleaned)
    cleaned = add_ratio_features(cleaned)

    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].map(
            lambda value: np.nan if pd.isna(value) else str(value).strip()
        )

    return cleaned
