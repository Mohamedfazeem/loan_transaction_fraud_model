# -*- coding: utf-8 -*-
"""
Fraud Detection Dashboard — Self-Contained Version
All data loading, cleaning, and model training happens
in-memory at startup (cached). No outputs/ folder needed.
CSVs must be in data/ folder committed to GitHub.
"""

import os, json, pathlib, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

warnings.filterwarnings("ignore")

# ── Resolve paths relative to this script ─────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
LOANS_CSV = DATA_DIR / "loan_applications.csv"
TXN_CSV   = DATA_DIR / "transactions.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

# ── Login ─────────────────────────────────────────────────────────────────────
USERS = {"admin": "1234"}
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if USERS.get(u) == p:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

# ── In-memory pipeline (runs once, then cached) ───────────────────────────────
@st.cache_data(show_spinner="⏳ Loading & cleaning data...")
def load_and_clean():
    if not LOANS_CSV.exists():
        st.error(
            f"❌ **`data/loan_applications.csv` not found.**\n\n"
            f"Looking in: `{LOANS_CSV}`\n\n"
            "Make sure the `data/` folder with both CSVs is committed to GitHub."
        )
        st.stop()

    # ── Clean loans ──────────────────────────────────────────────────────────
    loans = pd.read_csv(LOANS_CSV)
    loans["fraud_type"]        = loans["fraud_type"].fillna("None")
    loans["application_date"]  = pd.to_datetime(loans["application_date"])
    loans["loan_amount_requested"] = loans["loan_amount_requested"].abs()
    loans["monthly_income"]    = loans["monthly_income"].abs()
    for col in ["application_id","loan_type","employment_status","loan_status","gender"]:
        loans[col] = loans[col].astype(str).str.strip()
    loans = loans.sort_values("application_date", ascending=False)
    loans = loans.drop_duplicates(subset=["application_id"], keep="first")

    # ── Clean transactions ────────────────────────────────────────────────────
    txns = None
    if TXN_CSV.exists():
        txns = pd.read_csv(TXN_CSV)
        txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])
        for col in ["transaction_id","customer_id","transaction_status",
                    "transaction_type","merchant_category"]:
            if col in txns.columns:
                txns[col] = txns[col].astype(str).str.strip()
        if "transaction_location" in txns.columns:
            txns["State"] = txns["transaction_location"].astype(str)\
                              .str.split(",").str[-1].str.strip()

    return loans, txns


@st.cache_data(show_spinner="🤖 Training ML models (first load only)...")
def train_models():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    def encode(df):
        le = LabelEncoder()
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    def metrics_dict(name, y_true, y_pred, y_prob):
        r = classification_report(y_true, y_pred, output_dict=True)
        return {
            "model":     name,
            "accuracy":  round(r["accuracy"], 4),
            "precision": round(r.get("1", {}).get("precision", 0), 4),
            "recall":    round(r.get("1", {}).get("recall", 0), 4),
            "f1":        round(r.get("1", {}).get("f1-score", 0), 4),
            "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        }

    loans_raw = pd.read_csv(LOANS_CSV)
    loans_raw["fraud_type"] = loans_raw["fraud_type"].fillna("None")
    has_txn   = TXN_CSV.exists()
    txns_raw  = pd.read_csv(TXN_CSV) if has_txn else None

    # ── Model 1: Loan Fraud ──────────────────────────────────────────────────
    ldf = loans_raw.copy()
    ldf["application_date"] = pd.to_datetime(ldf["application_date"])
    ldf["app_year"]      = ldf["application_date"].dt.year
    ldf["app_month"]     = ldf["application_date"].dt.month
    ldf["app_day"]       = ldf["application_date"].dt.day
    ldf["app_dayofweek"] = ldf["application_date"].dt.dayofweek
    ldf.drop(columns=["application_date"], inplace=True)
    ldf = encode(ldf)
    X_l, y_l = ldf.drop("fraud_flag", axis=1), ldf["fraud_flag"]
    Xtr, Xte, ytr, yte = train_test_split(X_l, y_l, test_size=0.3, random_state=42)
    loan_model = RandomForestClassifier(n_estimators=200, random_state=42)
    loan_model.fit(Xtr, ytr)
    ypred = loan_model.predict(Xte)
    yprob = loan_model.predict_proba(Xte)[:, 1]
    loan_metrics = metrics_dict("Loan Fraud (RandomForest)", yte, ypred, yprob)
    all_loan_probs = loan_model.predict_proba(X_l)[:, 1]
    loan_pred = loans_raw[["application_id","customer_id","fraud_flag"]].copy()
    loan_pred["fraud_probability"] = (all_loan_probs * 100).round(2)
    loan_pred["predicted_fraud"]   = (all_loan_probs >= 0.5).astype(int)
    importances = pd.Series(loan_model.feature_importances_, index=X_l.columns)\
                    .sort_values(ascending=False).head(10)

    # ── Model 2: Transaction Fraud ───────────────────────────────────────────
    txn_metrics, txn_pred = None, None
    if has_txn:
        tdf = txns_raw.copy()
        tdf["transaction_date"] = pd.to_datetime(tdf["transaction_date"])
        tdf["trans_year"]      = tdf["transaction_date"].dt.year
        tdf["trans_month"]     = tdf["transaction_date"].dt.month
        tdf["trans_day"]       = tdf["transaction_date"].dt.day
        tdf["trans_hour"]      = tdf["transaction_date"].dt.hour
        tdf["trans_dayofweek"] = tdf["transaction_date"].dt.dayofweek
        tdf.drop(columns=["transaction_date"], inplace=True)
        tdf = encode(tdf)
        tdf.columns = tdf.columns.str.strip()
        X_t, y_t = tdf.drop("fraud_flag", axis=1), tdf["fraud_flag"]
        Xtr2, Xte2, ytr2, yte2 = train_test_split(X_t, y_t, test_size=0.3, random_state=42)
        txn_model = GradientBoostingClassifier(random_state=42)
        txn_model.fit(Xtr2, ytr2)
        ypred2 = txn_model.predict(Xte2)
        yprob2 = txn_model.predict_proba(Xte2)[:, 1]
        txn_metrics = metrics_dict("Transaction Fraud (GradientBoosting)", yte2, ypred2, yprob2)
        all_txn_probs = txn_model.predict_proba(X_t)[:, 1]
        txn_pred = txns_raw[["transaction_id","customer_id","fraud_flag"]].copy()
        txn_pred["fraud_probability"] = (all_txn_probs * 100).round(2)
        txn_pred["predicted_fraud"]   = (all_txn_probs >= 0.5).astype(int)

    # ── Model 3: Combined Risk ───────────────────────────────────────────────
    risk_df = None
    risk_metrics = None
    if has_txn:
        loan_enc = encode(ldf.copy())   # already encoded above
        tdf2 = txns_raw.copy()
        tdf2["transaction_date"] = pd.to_datetime(tdf2["transaction_date"])
        tdf2 = encode(tdf2.drop(columns=["transaction_date"]))
        tdf2.columns = tdf2.columns.str.strip()
        trans_agg = tdf2.groupby("customer_id").agg(
            total_transaction_amount=("transaction_amount", "sum"),
            avg_transaction_amount=("transaction_amount", "mean"),
            transaction_count=("transaction_amount", "count"),
            total_transaction_frauds=("fraud_flag", "sum"),
        ).reset_index()
        combined = ldf.merge(trans_agg, on="customer_id", how="left").fillna(0)
        X_c, y_c = combined.drop("fraud_flag", axis=1), combined["fraud_flag"]
        Xtr3, Xte3, ytr3, yte3 = train_test_split(X_c, y_c, test_size=0.3, random_state=42)
        risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        risk_model.fit(Xtr3, ytr3)
        ypred3 = risk_model.predict(Xte3)
        yprob3 = risk_model.predict_proba(Xte3)[:, 1]
        risk_metrics = metrics_dict("Combined Risk (RandomForest)", yte3, ypred3, yprob3)
        all_risk_probs = risk_model.predict_proba(X_c)[:, 1] * 100
        risk_df = pd.DataFrame({
            "customer_id_original": loans_raw["customer_id"].values,
            "risk_score":           all_risk_probs.round(2),
            "fraud_flag":           y_c.values,
        })

    all_metrics = [m for m in [loan_metrics, txn_metrics, risk_metrics] if m]
    return loan_pred, txn_pred, risk_df, all_metrics, importances


# ── Load everything ───────────────────────────────────────────────────────────
loan_df, txn_df                              = load_and_clean()
loan_pred, txn_pred, risk_df, model_metrics, importances = train_models()
has_txns = txn_df is not None
has_ml   = loan_pred is not None

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
pages = ["Executive Loan Portfolio",
         "Fraud Intelligence & Risk Mitigation",
         "Behavioral Risk Analysis",
         "🤖 ML Predictions & Model Performance"]
if not has_txns:
    pages.remove("Behavioral Risk Analysis")
page = st.sidebar.radio("Go to", pages)

st.sidebar.title("🔍 Filters")
loan_type_filter  = st.sidebar.multiselect("Loan Type",         loan_df["loan_type"].unique())
employment_filter = st.sidebar.multiselect("Employment Status", loan_df["employment_status"].unique())
gender_filter     = st.sidebar.multiselect("Gender",            loan_df["gender"].unique())
device_filter, state_filter = [], []
if has_txns:
    device_filter = st.sidebar.multiselect("Device", txn_df["device_used"].unique() if "device_used" in txn_df.columns else [])
    state_filter  = st.sidebar.multiselect("State",  txn_df["State"].unique()       if "State"       in txn_df.columns else [])
date_range = st.sidebar.date_input(
    "Application Date Range",
    [loan_df["application_date"].min(), loan_df["application_date"].max()])

# Apply filters
filtered_loans = loan_df.copy()
filtered_txns  = txn_df.copy() if has_txns else None
if loan_type_filter:  filtered_loans = filtered_loans[filtered_loans["loan_type"].isin(loan_type_filter)]
if employment_filter: filtered_loans = filtered_loans[filtered_loans["employment_status"].isin(employment_filter)]
if gender_filter:     filtered_loans = filtered_loans[filtered_loans["gender"].isin(gender_filter)]
if len(date_range) == 2:
    s, e = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_loans = filtered_loans[(filtered_loans["application_date"] >= s) &
                                    (filtered_loans["application_date"] <= e)]
if has_txns:
    if device_filter and "device_used" in filtered_txns.columns:
        filtered_txns = filtered_txns[filtered_txns["device_used"].isin(device_filter)]
    if state_filter and "State" in filtered_txns.columns:
        filtered_txns = filtered_txns[filtered_txns["State"].isin(state_filter)]


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Loan Portfolio
# ════════════════════════════════════════════════════════════════════════════
if page == "Executive Loan Portfolio":
    st.title("📘 Executive Loan Portfolio")
    total_apps         = len(filtered_loans)
    grand_total_amount = filtered_loans["loan_amount_requested"].sum()
    avg_cibil          = filtered_loans["cibil_score"].mean()
    approval_rate      = (filtered_loans["loan_status"] == "Approved").mean() * 100 if total_apps else 0
    avg_income         = filtered_loans["monthly_income"].mean()

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Applications", f"{total_apps:,}")
    m2.metric("Total Requested",    f"₹{grand_total_amount/1e7:.2f} Cr")
    m3.metric("Avg CIBIL Score",    f"{avg_cibil:.0f}")
    m4.metric("Approval Rate",      f"{approval_rate:.1f}%")
    m5.metric("Avg Monthly Income", f"₹{avg_income:,.0f}")
    st.markdown("---")

    demand = filtered_loans.groupby(["loan_type","loan_status"])["loan_amount_requested"].sum().reset_index()
    demand["pct"] = demand["loan_amount_requested"] / grand_total_amount * 100 if grand_total_amount else 0
    st.plotly_chart(px.bar(demand, x="loan_type", y="pct", color="loan_status", barmode="group",
        title="Demand by Loan Type (% of Grand Total)",
        text=demand["pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"pct":"% of Total","loan_type":"Loan Category"}
    ).update_traces(textposition="outside"), use_container_width=True)
    st.markdown("---")

    fl = filtered_loans.copy()
    fl["age_group"] = pd.cut(fl["applicant_age"],bins=[0,25,35,45,55,100],
                              labels=["18-25","26-35","36-45","46-55","56+"])
    age_df = fl["age_group"].value_counts().reindex(["18-25","26-35","36-45","46-55","56+"]).reset_index()
    age_df.columns = ["age_group","count"]
    age_df["pct"] = age_df["count"] / total_apps * 100 if total_apps else 0
    st.plotly_chart(px.bar(age_df, x="age_group", y="pct", color="age_group",
        title="Applicant Age Demographics",
        text=age_df["pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"age_group":"Age Group","pct":"% of Applicants"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    ).update_traces(textposition="outside"), use_container_width=True)
    st.markdown("---")

    status_df = filtered_loans["loan_status"].value_counts().reset_index()
    status_df.columns = ["loan_status","count"]
    st.plotly_chart(px.pie(status_df, values="count", names="loan_status",
        title="Approval Status Breakdown", hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Safe
    ).update_traces(textinfo="percent+label"), use_container_width=True)
    st.markdown("---")

    st.plotly_chart(px.scatter(filtered_loans, x="cibil_score", y="monthly_income",
        color="loan_status", opacity=0.5, title="CIBIL Score vs. Income",
        labels={"cibil_score":"CIBIL Score","monthly_income":"Monthly Income"}
    ), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Fraud Intelligence
# ════════════════════════════════════════════════════════════════════════════
elif page == "Fraud Intelligence & Risk Mitigation":
    st.title("🛡️ Fraud Intelligence & Risk Mitigation")

    def dti_cat(x):
        if x < 20: return "Excellent (<20)"
        elif x <= 35: return "Good (20–35)"
        elif x <= 40: return "Fair (36–40)"
        elif x <= 100: return "High Risk (40–100)"
        else: return "Out of Range (>100)"

    fl = filtered_loans.copy()
    fl["DTI_Risk_Category"] = fl["debt_to_income_ratio"].apply(dti_cat)
    fraud_loans   = fl[fl["fraud_flag"] == 1]
    total_fraud   = len(fraud_loans)
    loan_fraud_rt = total_fraud / len(fl) * 100 if len(fl) else 0
    undetected    = fl[fl["loan_status"] == "Fraudulent - Undetected"]
    undet_pct     = len(undetected) / total_fraud * 100 if total_fraud else 0
    risk_density  = total_fraud / fl["customer_id"].nunique() * 1000 if fl["customer_id"].nunique() else 0
    fraud_val     = fraud_loans["loan_amount_requested"].sum()

    if has_txns:
        fraud_txns   = filtered_txns[filtered_txns["fraud_flag"] == 1]
        total_fraud_rt = loan_fraud_rt + len(fraud_txns)/len(filtered_txns)*100
        total_fraud_val = fraud_val + fraud_txns["transaction_amount"].sum()
    else:
        total_fraud_rt  = loan_fraud_rt
        total_fraud_val = fraud_val

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Fraud Rate",        f"{total_fraud_rt:.2f}%")
    k2.metric("Total Fraudulent Value",  f"₹{total_fraud_val:,.0f}")
    k3.metric("Undetected Fraud (%)",    f"{undet_pct:.2f}%")
    k4.metric("Risk Density (per 1000)", f"{risk_density:.2f}")
    st.markdown("---")

    if len(fraud_loans) > 0:
        emp = fraud_loans.groupby(["employment_status","fraud_type"]).size().reset_index(name="count")
        emp["pct"] = emp.groupby("employment_status")["count"].transform(lambda x: x/x.sum()*100)
        st.plotly_chart(px.bar(emp, x="employment_status", y="pct", color="fraud_type",
            text=emp["pct"].round(1).astype(str)+"%",
            title="Fraud by Employment Status (%)"
        ).update_traces(textposition="inside"), use_container_width=True)
    st.markdown("---")

    dti_f = fl.groupby(["DTI_Risk_Category","fraud_flag"]).size().reset_index(name="count")
    dti_f["pct"] = dti_f.groupby("DTI_Risk_Category")["count"].transform(lambda x: x/x.sum()*100)
    st.plotly_chart(px.bar(dti_f, x="DTI_Risk_Category", y="pct", color="fraud_flag",
        text=dti_f["pct"].round(1).astype(str)+"%", title="DTI Risk Category vs Fraud (%)"
    ).update_traces(textposition="outside"), use_container_width=True)
    st.markdown("---")

    prop_f = fl.groupby(["property_ownership_status","fraud_flag"])["customer_id"].nunique().reset_index(name="count")
    prop_f["pct"] = prop_f.groupby("property_ownership_status")["count"].transform(lambda x: x/x.sum()*100)
    st.plotly_chart(px.bar(prop_f, x="property_ownership_status", y="pct", color="fraud_flag",
        barmode="group", text=prop_f["pct"].round(1).astype(str)+"%",
        title="Property Ownership vs Fraud (%)"
    ).update_traces(textposition="outside"), use_container_width=True)

    if has_txns:
        st.markdown("---")
        txn_amt = filtered_txns.groupby("fraud_flag")["transaction_amount"].sum().reset_index()
        txn_amt["label"] = txn_amt["fraud_flag"].map({0:"Legitimate",1:"Fraudulent"})
        st.plotly_chart(px.pie(txn_amt, names="label", values="transaction_amount",
            title="Transaction Amount: Fraud vs Legitimate", hole=0.4), use_container_width=True)
        st.markdown("---")
        if len(fraud_txns) > 0 and "State" in filtered_txns.columns:
            sf = fraud_txns["State"].value_counts(normalize=True).head(10).reset_index()
            sf.columns = ["State","pct"]; sf["pct"] *= 100
            st.plotly_chart(px.bar(sf, x="State", y="pct",
                text=sf["pct"].round(1).astype(str)+"%",
                title="Top 10 Fraudulent Transaction States (%)"
            ).update_traces(textposition="outside"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Behavioral Risk Analysis
# ════════════════════════════════════════════════════════════════════════════
elif page == "Behavioral Risk Analysis":
    st.title("🧠 Behavioral Risk Analysis")
    if not has_txns:
        st.warning("Transactions data not available.")
        st.stop()

    n = len(filtered_txns)
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total Transactions",    f"{n/1000:.1f}K")
    k2.metric("Total Customers",       f"{filtered_txns['customer_id'].nunique()/1000:.1f}K")
    k3.metric("Avg Transaction Value", f"₹{filtered_txns['transaction_amount'].mean():,.0f}")
    k4.metric("Success Rate",          f"{filtered_txns[filtered_txns['fraud_flag']==0].shape[0]/n*100:.1f}%")
    k5.metric("Top Location", filtered_txns["transaction_location"].value_counts().idxmax() if "transaction_location" in filtered_txns.columns else "N/A")
    k6.metric("Top Txn Type", filtered_txns["transaction_type"].value_counts().idxmax() if "transaction_type" in filtered_txns.columns else "N/A")
    st.markdown("---")

    c1,c2,c3 = st.columns(3)
    with c1:
        cat = filtered_txns.groupby("transaction_type")["transaction_amount"].sum().reset_index()
        st.plotly_chart(px.treemap(cat, path=["transaction_type"], values="transaction_amount",
            title="Spending by Transaction Type"
        ).update_traces(textinfo="label+percent entry"), use_container_width=True)
    with c2:
        if "device_used" in filtered_txns.columns:
            dev = filtered_txns.groupby(["device_used","fraud_flag"]).size().reset_index(name="count")
            dev["pct"] = dev.groupby("device_used")["count"].transform(lambda x: x/x.sum()*100)
            st.plotly_chart(px.bar(dev, x="device_used", y="pct", color="fraud_flag",
                text=dev["pct"].round(1).astype(str)+"%", title="Device Risk Analysis (%)"
            ).update_traces(textposition="outside"), use_container_width=True)
    with c3:
        if "is_international_transaction" in filtered_txns.columns:
            intl = filtered_txns.groupby("is_international_transaction")["transaction_amount"].sum().reset_index()
            intl["label"] = intl["is_international_transaction"].map({0:"Domestic",1:"International",False:"Domestic",True:"International"})
            intl["pct"]   = intl["transaction_amount"] / intl["transaction_amount"].sum() * 100
            st.plotly_chart(px.bar(intl, x="label", y="pct",
                text=intl["pct"].round(1).astype(str)+"%", title="International vs Domestic (%)"
            ).update_traces(textposition="outside"), use_container_width=True)

    st.markdown("---")
    vel = filtered_txns.groupby(filtered_txns["transaction_date"].dt.day)["transaction_amount"].sum().reset_index()
    vel.columns = ["day","amount"]
    vel["pct"] = vel["amount"] / vel["amount"].sum() * 100
    st.plotly_chart(px.area(vel, x="day", y="pct", title="Transaction Velocity by Day of Month (%)"), use_container_width=True)
    st.markdown("---")

    if "merchant_category" in filtered_txns.columns:
        mc = filtered_txns.groupby(["merchant_category","fraud_flag"]).size().reset_index(name="count")
        mc["pct"] = mc.groupby("merchant_category")["count"].transform(lambda x: x/x.sum()*100)
        mc_fraud = mc[mc["fraud_flag"]==1].sort_values("pct", ascending=False)
        st.plotly_chart(px.bar(mc_fraud, x="merchant_category", y="pct",
            text=mc_fraud["pct"].round(1).astype(str)+"%",
            title="Fraud Rate by Merchant Category (%)",
            color="pct", color_continuous_scale="Reds"
        ).update_traces(textposition="outside"), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(px.scatter(filtered_loans, x="monthly_income", y="loan_amount_requested",
        color="fraud_flag", opacity=0.5, title="Income vs Loan Amount (Fraud Highlighted)"
    ), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML Predictions
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Predictions & Model Performance":
    st.title("🤖 ML Predictions & Model Performance")

    # Model cards
    st.markdown("### 📊 Model Performance Summary")
    colors = ["#1f77b4","#ff7f0e","#2ca02c"]
    cols = st.columns(len(model_metrics))
    for i,(col,m) in enumerate(zip(cols, model_metrics)):
        with col:
            st.markdown(f"""
            <div style="background:{colors[i]};border-radius:10px;padding:16px;color:white;">
              <h4 style="margin:0 0 8px 0;">{m['model']}</h4>
              <p style="margin:2px 0;">Accuracy : <b>{m['accuracy']*100:.1f}%</b></p>
              <p style="margin:2px 0;">Precision: <b>{m['precision']*100:.1f}%</b></p>
              <p style="margin:2px 0;">Recall   : <b>{m['recall']*100:.1f}%</b></p>
              <p style="margin:2px 0;">F1 Score : <b>{m['f1']*100:.1f}%</b></p>
              <p style="margin:2px 0;">ROC-AUC  : <b>{m['roc_auc']:.4f}</b></p>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # Comparison bar chart
    rows = []
    for m in model_metrics:
        for mn in ["accuracy","precision","recall","f1","roc_auc"]:
            rows.append({"Model":m["model"],"Metric":mn.upper(),"Score":round(m[mn],4)})
    comp_df = pd.DataFrame(rows)
    st.plotly_chart(px.bar(comp_df, x="Metric", y="Score", color="Model", barmode="group",
        title="Model Performance Comparison",
        text=comp_df["Score"].apply(lambda x: f"{x:.3f}")
    ).update_traces(textposition="outside").update_yaxes(range=[0,1.15]),
    use_container_width=True)

    # Feature importances
    st.markdown("---")
    st.markdown("### 🔍 Top 10 Feature Importances — Loan Fraud Model")
    fi_df = importances.reset_index()
    fi_df.columns = ["Feature","Importance"]
    st.plotly_chart(px.bar(fi_df, x="Importance", y="Feature", orientation="h",
        title="Feature Importances", color="Importance", color_continuous_scale="Blues"
    ), use_container_width=True)
    st.markdown("---")

    # Loan predictions
    st.markdown("### 🏦 Loan Fraud Predictions")
    lp = loan_pred.copy()
    lp_f = lp[lp["application_id"].isin(filtered_loans["application_id"])] if len(filtered_loans) < len(loan_df) else lp
    l1,l2,l3 = st.columns(3)
    l1.metric("Loans Scored",         f"{len(lp_f):,}")
    l2.metric("Predicted Fraudulent", f"{lp_f['predicted_fraud'].sum():,}")
    l3.metric("Predicted Fraud Rate", f"{lp_f['predicted_fraud'].mean()*100:.2f}%")
    st.plotly_chart(px.histogram(lp_f, x="fraud_probability", color="predicted_fraud", nbins=50,
        title="Loan Fraud Probability Distribution",
        labels={"fraud_probability":"Fraud Probability (%)","predicted_fraud":"Predicted"}
    ), use_container_width=True)
    st.markdown("#### ⚠️ Top 20 Highest-Risk Loan Applications")
    top20 = lp_f.sort_values("fraud_probability",ascending=False).head(20).merge(
        loan_df[["application_id","loan_type","loan_amount_requested","employment_status","cibil_score","loan_status"]],
        on="application_id", how="left")
    top20["fraud_probability"] = top20["fraud_probability"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(top20[["application_id","loan_type","loan_amount_requested",
                         "employment_status","cibil_score","loan_status",
                         "fraud_probability","predicted_fraud"]], use_container_width=True)
    st.markdown("---")

    # Transaction predictions
    if txn_pred is not None and has_txns:
        st.markdown("### 💳 Transaction Fraud Predictions")
        tp_f = txn_pred.copy()
        t1,t2,t3 = st.columns(3)
        t1.metric("Transactions Scored",  f"{len(tp_f):,}")
        t2.metric("Predicted Fraudulent", f"{tp_f['predicted_fraud'].sum():,}")
        t3.metric("Predicted Fraud Rate", f"{tp_f['predicted_fraud'].mean()*100:.2f}%")
        st.plotly_chart(px.histogram(tp_f, x="fraud_probability", color="predicted_fraud", nbins=50,
            title="Transaction Fraud Probability Distribution",
            labels={"fraud_probability":"Fraud Probability (%)","predicted_fraud":"Predicted"}
        ), use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            ad = tp_f["fraud_flag"].value_counts().reset_index()
            ad.columns = ["fraud_flag","count"]
            ad["label"] = ad["fraud_flag"].map({0:"Legitimate",1:"Fraudulent"})
            st.plotly_chart(px.pie(ad, values="count", names="label", title="Actual Fraud Distribution", hole=0.4), use_container_width=True)
        with c2:
            pd2 = tp_f["predicted_fraud"].value_counts().reset_index()
            pd2.columns = ["predicted_fraud","count"]
            pd2["label"] = pd2["predicted_fraud"].map({0:"Legitimate",1:"Fraudulent"})
            st.plotly_chart(px.pie(pd2, values="count", names="label", title="Predicted Fraud Distribution", hole=0.4), use_container_width=True)
        st.markdown("#### ⚠️ Top 20 Highest-Risk Transactions")
        top20t = tp_f.sort_values("fraud_probability",ascending=False).head(20).merge(
            txn_df[["transaction_id","transaction_type","transaction_amount",
                    "merchant_category","device_used","transaction_status"]],
            on="transaction_id", how="left")
        top20t["fraud_probability"] = top20t["fraud_probability"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(top20t[["transaction_id","transaction_type","transaction_amount",
                              "merchant_category","device_used","transaction_status",
                              "fraud_probability","predicted_fraud"]], use_container_width=True)
        st.markdown("---")

    # Risk scores
    if risk_df is not None:
        st.markdown("### 👤 Customer Risk Scores (Combined Model)")
        rd = risk_df.copy()
        r1,r2,r3 = st.columns(3)
        r1.metric("Customers Scored", f"{len(rd):,}")
        r2.metric("High Risk (≥50%)", f"{len(rd[rd['risk_score']>=50]):,}")
        r3.metric("Avg Risk Score",   f"{rd['risk_score'].mean():.1f}%")
        fig_risk = px.histogram(rd, x="risk_score", nbins=50, title="Customer Risk Score Distribution",
            labels={"risk_score":"Risk Score (%)"}, color_discrete_sequence=["#e74c3c"])
        fig_risk.add_vline(x=50, line_dash="dash", line_color="black", annotation_text="High Risk Threshold")
        st.plotly_chart(fig_risk, use_container_width=True)
        rd["risk_band"] = pd.cut(rd["risk_score"],bins=[0,20,40,60,80,100],
            labels=["Very Low (0-20)","Low (20-40)","Medium (40-60)","High (60-80)","Critical (80-100)"])
        rb = rd["risk_band"].value_counts().reset_index(); rb.columns=["risk_band","count"]
        st.plotly_chart(px.bar(rb, x="risk_band", y="count", title="Customers by Risk Band",
            color="risk_band",
            color_discrete_sequence=["#2ecc71","#f39c12","#e67e22","#e74c3c","#8e44ad"]
        ), use_container_width=True)
        st.markdown("#### 🚨 Top 20 Highest Risk Customers")
        tr = rd.sort_values("risk_score",ascending=False).head(20)[["customer_id_original","risk_score","fraud_flag"]].copy()
        tr.columns = ["Customer ID","Risk Score (%)","Actual Fraud"]
        tr["Risk Score (%)"] = tr["Risk Score (%)"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(tr, use_container_width=True)
