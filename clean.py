"""
clean.py — Data Cleaning Pipeline
Mirrors the logic in data_cleaning.sql using pandas.
"""
import pandas as pd
import os

RAW_LOANS_PATH = "data/loan_applications.csv"
RAW_TXN_PATH   = "data/transactions.csv"
OUT_LOANS_PATH = "outputs/loan_applications_filtered.csv"
OUT_TXN_PATH   = "outputs/transactions_cleaned.csv"


def clean_loans(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[clean] Loans raw rows: {len(df)}")
    df = df.copy()
    df["fraud_type"] = df["fraud_type"].fillna("None")
    for col in ["application_id", "loan_type", "employment_status", "loan_status", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["application_date"] = pd.to_datetime(df["application_date"])
    df = df.sort_values("application_date", ascending=False)
    df = df.drop_duplicates(subset=["application_id"], keep="first")
    df["loan_amount_requested"] = df["loan_amount_requested"].abs()
    df["monthly_income"]        = df["monthly_income"].abs()
    outliers = len(df[df["debt_to_income_ratio"] > 100])
    if outliers:
        print(f"[clean]   DTI > 100 flagged: {outliers} rows")
    print(f"[clean] Loans after cleaning: {len(df)}")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[clean] Transactions raw rows: {len(df)}")
    df = df.copy()
    for col in ["transaction_id", "customer_id", "transaction_status",
                "transaction_type", "merchant_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    if "transaction_location" in df.columns:
        df["State"] = df["transaction_location"].astype(str).str.split(",").str[-1].str.strip()
    print(f"[clean] Transactions after cleaning: {len(df)}")
    return df


def run_cleaning() -> dict:
    results = {}
    loans_raw   = pd.read_csv(RAW_LOANS_PATH)
    loans_clean = clean_loans(loans_raw)
    loans_clean.to_csv(OUT_LOANS_PATH, index=False)
    print(f"[clean] ✓ Loans saved → {OUT_LOANS_PATH}")
    results["loans"] = loans_clean

    if os.path.exists(RAW_TXN_PATH):
        txn_raw   = pd.read_csv(RAW_TXN_PATH)
        txn_clean = clean_transactions(txn_raw)
        txn_clean.to_csv(OUT_TXN_PATH, index=False)
        print(f"[clean] ✓ Transactions saved → {OUT_TXN_PATH}")
        results["transactions"] = txn_clean
    else:
        print(f"[clean] ⚠ transactions.csv not found in data/ — skipping.")
        results["transactions"] = None

    return results


if __name__ == "__main__":
    run_cleaning()
