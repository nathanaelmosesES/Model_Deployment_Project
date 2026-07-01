from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "Credit_Score"
ID_COLUMNS = ["", "ID", "Customer_ID", "Name", "SSN", "Month"]

NUMERIC_FEATURES = [
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
    "Credit_History_Age_Months",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Loan_Type_Count",
    "Has_Auto_Loan",
    "Has_Credit_Builder_Loan",
    "Has_Debt_Consolidation_Loan",
    "Has_Home_Equity_Loan",
    "Has_Mortgage_Loan",
    "Has_Payday_Loan",
    "Has_Personal_Loan",
    "Has_Student_Loan",
    "Debt_To_Income",
    "EMI_To_Salary",
    "Investment_To_Salary",
    "Balance_To_Salary",
]

CATEGORICAL_FEATURES = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "Month",
]

RAW_NUMERIC_COLUMNS = [
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
