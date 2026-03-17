"""
model.py — Fraud Detection ML Pipeline
Three models from ML_modal.ipynb:
  1. Loan Fraud Model       — RandomForestClassifier
  2. Transaction Fraud Model — GradientBoostingClassifier
  3. Combined Risk Model    — RandomForestClassifier (loan + transaction aggregates)

Outputs:
  outputs/loan_predictions.csv
  outputs/transaction_predictions.csv
  outputs/risk_scores.csv
  outputs/model_metrics.json
"""

import json
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

LOANS_PATH = "outputs/loan_applications_filtered.csv"
TXN_PATH   = "outputs/transactions_cleaned.csv"


# ─── helpers ────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def metrics_dict(name, y_true, y_pred, y_prob) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "model": name,
        "accuracy":  round(report["accuracy"], 4),
        "precision": round(report.get("1", {}).get("precision", 0), 4),
        "recall":    round(report.get("1", {}).get("recall", 0), 4),
        "f1":        round(report.get("1", {}).get("f1-score", 0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


# ─── Model 1: Loan Fraud ────────────────────────────────────────────────────

def train_loan_model(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df["application_date"] = pd.to_datetime(df["application_date"])
    df["app_year"]       = df["application_date"].dt.year
    df["app_month"]      = df["application_date"].dt.month
    df["app_day"]        = df["application_date"].dt.day
    df["app_dayofweek"]  = df["application_date"].dt.dayofweek
    df.drop(columns=["application_date"], inplace=True)
    df = encode_categoricals(df)

    X = df.drop("fraud_flag", axis=1)
    y = df["fraud_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = metrics_dict("Loan Fraud (RandomForest)", y_test, y_pred, y_prob)
    print(f"[model] Loan Fraud — Accuracy: {metrics['accuracy']}  ROC-AUC: {metrics['roc_auc']}")

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns) \
                    .sort_values(ascending=False).head(10)

    # Predictions on full dataset
    all_probs = model.predict_proba(df.drop("fraud_flag", axis=1))[:, 1]
    return model, metrics, importances, all_probs


# ─── Model 2: Transaction Fraud ─────────────────────────────────────────────

def train_transaction_model(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["trans_year"]      = df["transaction_date"].dt.year
    df["trans_month"]     = df["transaction_date"].dt.month
    df["trans_day"]       = df["transaction_date"].dt.day
    df["trans_hour"]      = df["transaction_date"].dt.hour
    df["trans_dayofweek"] = df["transaction_date"].dt.dayofweek
    df.drop(columns=["transaction_date"], inplace=True)

    # Drop State column if present (added by clean.py, not a model feature)
    if "State" in df.columns:
        df.drop(columns=["State"], inplace=True)

    df = encode_categoricals(df)
    df.columns = df.columns.str.strip()

    X = df.drop("fraud_flag", axis=1)
    y = df["fraud_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = metrics_dict("Transaction Fraud (GradientBoosting)", y_test, y_pred, y_prob)
    print(f"[model] Txn Fraud — Accuracy: {metrics['accuracy']}  ROC-AUC: {metrics['roc_auc']}")

    all_probs = model.predict_proba(df.drop("fraud_flag", axis=1))[:, 1]
    return model, metrics, all_probs, X.columns.tolist()


# ─── Model 3: Combined Customer Risk ────────────────────────────────────────

def train_risk_model(loans_raw: pd.DataFrame, txn_raw: pd.DataFrame):
    # Encode loans
    loan_df = loans_raw.copy()
    loan_df["application_date"] = pd.to_datetime(loan_df["application_date"])
    loan_df["app_year"]      = loan_df["application_date"].dt.year
    loan_df["app_month"]     = loan_df["application_date"].dt.month
    loan_df["app_day"]       = loan_df["application_date"].dt.day
    loan_df["app_dayofweek"] = loan_df["application_date"].dt.dayofweek
    loan_df.drop(columns=["application_date"], inplace=True)
    loan_df = encode_categoricals(loan_df)

    # Aggregate transactions per customer
    txn_df = txn_raw.copy()
    txn_df["transaction_date"] = pd.to_datetime(txn_df["transaction_date"])
    le = LabelEncoder()
    for col in txn_df.select_dtypes(include=["object"]).columns:
        txn_df[col] = le.fit_transform(txn_df[col].astype(str))
    if "State" in txn_df.columns:
        txn_df.drop(columns=["State"], inplace=True)

    trans_agg = txn_df.groupby("customer_id").agg(
        total_transaction_amount=("transaction_amount", "sum"),
        avg_transaction_amount=("transaction_amount", "mean"),
        transaction_count=("transaction_amount", "count"),
        total_transaction_frauds=("fraud_flag", "sum"),
    ).reset_index()

    combined_df = loan_df.merge(trans_agg, on="customer_id", how="left").fillna(0)

    X = combined_df.drop("fraud_flag", axis=1)
    y = combined_df["fraud_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = metrics_dict("Combined Risk (RandomForest)", y_test, y_pred, y_prob)
    print(f"[model] Combined Risk — Accuracy: {metrics['accuracy']}  ROC-AUC: {metrics['roc_auc']}")

    # Risk scores for all customers
    all_probs = model.predict_proba(X)[:, 1] * 100
    risk_df = combined_df[["customer_id"]].copy()
    risk_df = risk_df.reset_index(drop=True)

    # Re-decode customer_id — get original IDs from raw loans
    cust_ids = loans_raw["customer_id"].values
    risk_df["customer_id_original"] = cust_ids
    risk_df["risk_score"] = all_probs
    risk_df["fraud_flag"] = y.values

    return model, metrics, risk_df


# ─── Main entry point ───────────────────────────────────────────────────────

def run_models() -> dict:
    print("[model] Loading cleaned data...")
    loans_df = pd.read_csv(LOANS_PATH)
    txn_df   = pd.read_csv(TXN_PATH)

    results = {}

    # 1. Loan fraud model
    print("[model] Training Loan Fraud model...")
    loan_model, loan_metrics, loan_importances, loan_probs = train_loan_model(loans_df)
    loan_out = loans_df[["application_id", "customer_id", "fraud_flag"]].copy()
    loan_out["fraud_probability"] = (loan_probs * 100).round(2)
    loan_out["predicted_fraud"]   = (loan_probs >= 0.5).astype(int)
    loan_out.to_csv("outputs/loan_predictions.csv", index=False)
    results["loan_metrics"] = loan_metrics
    results["loan_importances"] = loan_importances.to_dict()

    # 2. Transaction fraud model
    print("[model] Training Transaction Fraud model...")
    txn_model, txn_metrics, txn_probs, txn_features = train_transaction_model(txn_df)
    txn_out = txn_df[["transaction_id", "customer_id", "fraud_flag"]].copy()
    txn_out["fraud_probability"] = (txn_probs * 100).round(2)
    txn_out["predicted_fraud"]   = (txn_probs >= 0.5).astype(int)
    txn_out.to_csv("outputs/transaction_predictions.csv", index=False)
    results["txn_metrics"] = txn_metrics

    # 3. Combined risk model
    print("[model] Training Combined Risk model...")
    risk_model, risk_metrics, risk_df = train_risk_model(loans_df, txn_df)
    risk_df.to_csv("outputs/risk_scores.csv", index=False)
    results["risk_metrics"] = risk_metrics

    # Save all metrics to JSON
    all_metrics = [loan_metrics, txn_metrics, risk_metrics]
    with open("outputs/model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("[model] ✓ All models trained. Outputs saved to outputs/")
    return results


if __name__ == "__main__":
    run_models()
