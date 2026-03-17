# -*- coding: utf-8 -*-
"""
Fraud Detection Dashboard — Full Version
Pages:
  1. Executive Loan Portfolio
  2. Fraud Intelligence & Risk Mitigation
  3. Behavioral Risk Analysis
  4. ML Predictions & Model Performance
"""

import os, json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ── Login ────────────────────────────────────────────────────────────────────
USERS = {"admin": "1234"}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if USERS.get(username) == password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout():
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

if not st.session_state["authenticated"]:
    login()
    st.stop()

logout()

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=200)

# ── Data paths ───────────────────────────────────────────────────────────────
LOANS_PATH    = "outputs/loan_applications_filtered.csv"
TXN_PATH      = "outputs/transactions_cleaned.csv"
LOAN_PRED     = "outputs/loan_predictions.csv"
TXN_PRED      = "outputs/transaction_predictions.csv"
RISK_SCORES   = "outputs/risk_scores.csv"
METRICS_PATH  = "outputs/model_metrics.json"

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(LOANS_PATH):
        st.error("❌ Run `python pipeline_runner.py` first to generate cleaned data.")
        st.stop()

    loans = pd.read_csv(LOANS_PATH)
    loans["application_date"] = pd.to_datetime(loans["application_date"])
    loans["fraud_type"] = loans["fraud_type"].fillna("None")

    txns = None
    if os.path.exists(TXN_PATH):
        txns = pd.read_csv(TXN_PATH)
        txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])
        if "transaction_location" in txns.columns:
            txns["State"] = txns["transaction_location"].astype(str).str.split(",").str[-1].str.strip()

    return loans, txns

@st.cache_data
def load_ml_outputs():
    loan_pred, txn_pred, risk_df, metrics = None, None, None, []
    if os.path.exists(LOAN_PRED):
        loan_pred = pd.read_csv(LOAN_PRED)
    if os.path.exists(TXN_PRED):
        txn_pred = pd.read_csv(TXN_PRED)
    if os.path.exists(RISK_SCORES):
        risk_df = pd.read_csv(RISK_SCORES)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return loan_pred, txn_pred, risk_df, metrics

loan_df, txn_df = load_data()
has_txns = txn_df is not None
loan_pred, txn_pred, risk_df, model_metrics = load_ml_outputs()
has_ml = loan_pred is not None

# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
pages = [
    "Executive Loan Portfolio",
    "Fraud Intelligence & Risk Mitigation",
    "Behavioral Risk Analysis",
    "🤖 ML Predictions & Model Performance",
]
if not has_txns:
    pages.remove("Behavioral Risk Analysis")

page = st.sidebar.radio("Go to", pages)



# ── Filters ──────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")
loan_type_filter  = st.sidebar.multiselect("Loan Type",         loan_df["loan_type"].unique())
employment_filter = st.sidebar.multiselect("Employment Status", loan_df["employment_status"].unique())
gender_filter     = st.sidebar.multiselect("Gender",            loan_df["gender"].unique())

if has_txns:
    device_filter = st.sidebar.multiselect("Device",
        txn_df["device_used"].unique() if "device_used" in txn_df.columns else [])
    state_filter  = st.sidebar.multiselect("State", txn_df["State"].unique() if "State" in txn_df.columns else [])

date_range = st.sidebar.date_input(
    "Application Date Range",
    [loan_df["application_date"].min(), loan_df["application_date"].max()]
)

# Apply filters
filtered_loans = loan_df.copy()
filtered_txns  = txn_df.copy() if has_txns else None

if loan_type_filter:
    filtered_loans = filtered_loans[filtered_loans["loan_type"].isin(loan_type_filter)]
if employment_filter:
    filtered_loans = filtered_loans[filtered_loans["employment_status"].isin(employment_filter)]
if gender_filter:
    filtered_loans = filtered_loans[filtered_loans["gender"].isin(gender_filter)]
if len(date_range) == 2:
    s, e = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_loans = filtered_loans[(filtered_loans["application_date"] >= s) &
                                    (filtered_loans["application_date"] <= e)]
if has_txns:
    if device_filter and "device_used" in filtered_txns.columns:
        filtered_txns = filtered_txns[filtered_txns["device_used"].isin(device_filter)]
    if state_filter and "State" in filtered_txns.columns:
        filtered_txns = filtered_txns[filtered_txns["State"].isin(state_filter)]

# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Loan Portfolio
# ════════════════════════════════════════════════════════════════════════
if page == "Executive Loan Portfolio":
    st.title("📘 Executive Loan Portfolio")
    st.markdown("### High-Level Performance Metrics")

    total_apps         = len(filtered_loans)
    grand_total_amount = filtered_loans["loan_amount_requested"].sum()
    avg_cibil          = filtered_loans["cibil_score"].mean()
    approval_rate      = (filtered_loans["loan_status"] == "Approved").mean() * 100 if total_apps else 0
    avg_income         = filtered_loans["monthly_income"].mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Applications",  f"{total_apps:,}")
    m2.metric("Total Requested",     f"₹{grand_total_amount/1e7:.2f} Cr")
    m3.metric("Avg CIBIL Score",     f"{avg_cibil:.0f}")
    m4.metric("Approval Rate",       f"{approval_rate:.1f}%")
    m5.metric("Avg Monthly Income",  f"₹{avg_income:,.0f}")
    st.markdown("---")

    # 1. Demand by Loan Type
    demand = (filtered_loans.groupby(["loan_type", "loan_status"])["loan_amount_requested"]
              .sum().reset_index())
    demand["pct"] = demand["loan_amount_requested"] / grand_total_amount * 100 if grand_total_amount else 0
    fig = px.bar(demand, x="loan_type", y="pct", color="loan_status", barmode="group",
                 title="Demand by Loan Type (% of Grand Total Amount)",
                 text=demand["pct"].apply(lambda x: f"{x:.1f}%"),
                 labels={"pct": "% of Total", "loan_type": "Loan Category"})
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # 2. Age Demographics
    fl = filtered_loans.copy()
    fl["age_group"] = pd.cut(fl["applicant_age"],
                             bins=[0,25,35,45,55,100],
                             labels=["18-25","26-35","36-45","46-55","56+"])
    age_df = fl["age_group"].value_counts().reindex(["18-25","26-35","36-45","46-55","56+"]).reset_index()
    age_df.columns = ["age_group","count"]
    age_df["pct"] = age_df["count"] / total_apps * 100 if total_apps else 0
    fig2 = px.bar(age_df, x="age_group", y="pct", color="age_group",
                  title="Applicant Age Demographics (% of Total Applicants)",
                  text=age_df["pct"].apply(lambda x: f"{x:.1f}%"),
                  labels={"age_group":"Age Group","pct":"% of Applicants"},
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    fig2.update_traces(textposition="outside")
    fig2.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

    # 3. Approval Status Donut
    status_df = filtered_loans["loan_status"].value_counts().reset_index()
    status_df.columns = ["loan_status","count"]
    fig3 = px.pie(status_df, values="count", names="loan_status",
                  title="Approval Status Breakdown", hole=0.5,
                  color_discrete_sequence=px.colors.qualitative.Safe)
    fig3.update_traces(textinfo="percent+label")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")

    # 4. CIBIL vs Income scatter
    fig4 = px.scatter(filtered_loans, x="cibil_score", y="monthly_income",
                      color="loan_status", opacity=0.5,
                      title="CIBIL Score vs. Income Analysis",
                      labels={"cibil_score":"CIBIL Score","monthly_income":"Monthly Income"})
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE 2 — Fraud Intelligence & Risk Mitigation
# ════════════════════════════════════════════════════════════════════════
elif page == "Fraud Intelligence & Risk Mitigation":
    st.title("🛡️ Fraud Intelligence & Risk Mitigation")
    st.markdown("### High-Risk Activity & Pattern Analysis")

    def dti_category(x):
        if x < 20:          return "Excellent (<20)"
        elif x <= 35:       return "Good (20–35)"
        elif x <= 40:       return "Fair (36–40)"
        elif x <= 100:      return "High Risk (40–100)"
        else:               return "Out of Range (>100)"

    fl = filtered_loans.copy()
    fl["DTI_Risk_Category"] = fl["debt_to_income_ratio"].apply(dti_category)
    fraud_loans = fl[fl["fraud_flag"] == 1]
    total_fraud = len(fraud_loans)
    loan_fraud_rate = total_fraud / len(fl) * 100 if len(fl) else 0
    undetected = fl[fl["loan_status"] == "Fraudulent - Undetected"]
    undetected_pct = len(undetected) / total_fraud * 100 if total_fraud else 0
    total_customers = fl["customer_id"].nunique()
    risk_density = (total_fraud / total_customers) * 1000 if total_customers else 0
    fraud_value_loans = fraud_loans["loan_amount_requested"].sum()

    if has_txns:
        fraud_txns = filtered_txns[filtered_txns["fraud_flag"] == 1]
        txn_fraud_rate = len(fraud_txns) / len(filtered_txns) * 100 if len(filtered_txns) else 0
        total_fraud_rate = loan_fraud_rate + txn_fraud_rate
        total_fraud_value = fraud_value_loans + fraud_txns["transaction_amount"].sum()
    else:
        total_fraud_rate = loan_fraud_rate
        total_fraud_value = fraud_value_loans

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Fraud Rate",        f"{total_fraud_rate:.2f}%")
    k2.metric("Total Fraudulent Value",  f"₹{total_fraud_value:,.0f}")
    k3.metric("Undetected Fraud (%)",    f"{undetected_pct:.2f}%")
    k4.metric("Risk Density (per 1000)", f"{risk_density:.2f}")
    st.markdown("---")

    # 1. Fraud by Employment Status
    if len(fraud_loans) > 0:
        emp_fraud = (fraud_loans.groupby(["employment_status","fraud_type"])
                     .size().reset_index(name="count"))
        emp_fraud["pct"] = emp_fraud.groupby("employment_status")["count"].transform(
            lambda x: x / x.sum() * 100)
        fig = px.bar(emp_fraud, x="employment_status", y="pct", color="fraud_type",
                     text=emp_fraud["pct"].round(1).astype(str)+"%",
                     title="Fraud by Employment Status (%)",
                     labels={"pct":"% Loan Fraud","employment_status":"Employment Status"})
        fig.update_traces(textposition="inside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2. DTI Risk Category vs Fraud
    dti_fraud = fl.groupby(["DTI_Risk_Category","fraud_flag"]).size().reset_index(name="count")
    dti_fraud["pct"] = dti_fraud.groupby("DTI_Risk_Category")["count"].transform(
        lambda x: x / x.sum() * 100)
    fig2 = px.bar(dti_fraud, x="DTI_Risk_Category", y="pct", color="fraud_flag",
                  text=dti_fraud["pct"].round(1).astype(str)+"%",
                  title="DTI Risk Category vs Fraud (%)",
                  labels={"pct":"% of Applications","DTI_Risk_Category":"DTI Category",
                          "fraud_flag":"Fraud Flag"})
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

    # 3. Property Ownership vs Fraud
    prop_fraud = (fl.groupby(["property_ownership_status","fraud_flag"])["customer_id"]
                  .nunique().reset_index(name="count"))
    prop_fraud["pct"] = prop_fraud.groupby("property_ownership_status")["count"].transform(
        lambda x: x / x.sum() * 100)
    fig3 = px.bar(prop_fraud, x="property_ownership_status", y="pct", color="fraud_flag",
                  barmode="group", text=prop_fraud["pct"].round(1).astype(str)+"%",
                  title="Property Ownership vs Fraud (%)",
                  labels={"pct":"% Distinct Customers","property_ownership_status":"Ownership",
                          "fraud_flag":"Fraud Flag"})
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)

    if has_txns:
        st.markdown("---")
        # 4. Transaction Amount Fraud Pie
        txn_amt = (filtered_txns.groupby("fraud_flag")["transaction_amount"]
                   .sum().reset_index())
        txn_amt["label"] = txn_amt["fraud_flag"].map({0:"Legitimate",1:"Fraudulent"})
        fig4 = px.pie(txn_amt, names="label", values="transaction_amount",
                      title="Transaction Amount: Fraud vs Legitimate", hole=0.4)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("---")

        # 5. Top 10 Fraudulent States
        if len(fraud_txns) > 0 and "State" in filtered_txns.columns:
            state_fraud = (fraud_txns["State"].value_counts(normalize=True)
                           .head(10).reset_index())
            state_fraud.columns = ["State","pct"]
            state_fraud["pct"] *= 100
            fig5 = px.bar(state_fraud, x="State", y="pct",
                          text=state_fraud["pct"].round(1).astype(str)+"%",
                          title="Top 10 Fraudulent Transaction States (%)",
                          labels={"pct":"% Fraud Transactions"})
            fig5.update_traces(textposition="outside")
            st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — Behavioral Risk Analysis
# ════════════════════════════════════════════════════════════════════════
elif page == "Behavioral Risk Analysis":
    st.title("🧠 Behavioral Risk Analysis")

    if not has_txns:
        st.warning("Transactions data not available. Add data/transactions.csv and re-run pipeline.")
        st.stop()

    total_transactions = len(filtered_txns)
    total_customers    = filtered_txns["customer_id"].nunique()
    avg_txn_value      = filtered_txns["transaction_amount"].mean()
    success_rate       = (filtered_txns[filtered_txns["fraud_flag"] == 0].shape[0]
                          / total_transactions * 100) if total_transactions else 0
    top_location = (filtered_txns["transaction_location"].value_counts().idxmax()
                    if "transaction_location" in filtered_txns.columns else "N/A")
    top_txn_type = (filtered_txns["transaction_type"].value_counts().idxmax()
                    if "transaction_type" in filtered_txns.columns else "N/A")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Transactions",    f"{total_transactions/1000:.1f}K")
    k2.metric("Total Customers",       f"{total_customers/1000:.1f}K")
    k3.metric("Avg Transaction Value", f"₹{avg_txn_value:,.0f}")
    k4.metric("Success Rate",          f"{success_rate:.1f}%")
    k5.metric("Top Location",          top_location)
    k6.metric("Top Txn Type",          top_txn_type)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        cat_spend = (filtered_txns.groupby("transaction_type")["transaction_amount"]
                     .sum().reset_index())
        fig = px.treemap(cat_spend, path=["transaction_type"], values="transaction_amount",
                         title="Spending by Transaction Type")
        fig.update_traces(textinfo="label+percent entry")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "device_used" in filtered_txns.columns:
            dev_risk = (filtered_txns.groupby(["device_used","fraud_flag"])
                        .size().reset_index(name="count"))
            dev_risk["pct"] = dev_risk.groupby("device_used")["count"].transform(
                lambda x: x / x.sum() * 100)
            fig2 = px.bar(dev_risk, x="device_used", y="pct", color="fraud_flag",
                          text=dev_risk["pct"].round(1).astype(str)+"%",
                          title="Device Risk Analysis (%)",
                          labels={"pct":"Percentage (%)","device_used":"Device",
                                  "fraud_flag":"Fraud Flag"})
            fig2.update_traces(textposition="outside")
            fig2.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig2, use_container_width=True)

    with c3:
        if "is_international_transaction" in filtered_txns.columns:
            intl = (filtered_txns.groupby("is_international_transaction")["transaction_amount"]
                    .sum().reset_index())
            intl["label"] = intl["is_international_transaction"].map(
                {0:"Domestic",1:"International",False:"Domestic",True:"International"})
            intl["pct"] = intl["transaction_amount"] / intl["transaction_amount"].sum() * 100
            fig3 = px.bar(intl, x="label", y="pct",
                          text=intl["pct"].round(1).astype(str)+"%",
                          title="International vs Domestic (%)",
                          labels={"pct":"% of Amount","label":"Type"})
            fig3.update_traces(textposition="outside")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Transaction Velocity
    vel = (filtered_txns.groupby(filtered_txns["transaction_date"].dt.day)["transaction_amount"]
           .sum().reset_index())
    vel.columns = ["day_of_month","transaction_amount"]
    vel["pct"] = vel["transaction_amount"] / vel["transaction_amount"].sum() * 100
    fig4 = px.area(vel, x="day_of_month", y="pct",
                   title="Transaction Velocity by Day of Month (%)",
                   labels={"pct":"% of Monthly Amount","day_of_month":"Day"})
    fig4.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Merchant Category Fraud Breakdown
    if "merchant_category" in filtered_txns.columns:
        merch = (filtered_txns.groupby(["merchant_category","fraud_flag"])
                 .size().reset_index(name="count"))
        merch["pct"] = merch.groupby("merchant_category")["count"].transform(
            lambda x: x / x.sum() * 100)
        fraud_merch = merch[merch["fraud_flag"] == 1].sort_values("pct", ascending=False)
        fig5 = px.bar(fraud_merch, x="merchant_category", y="pct",
                      text=fraud_merch["pct"].round(1).astype(str)+"%",
                      title="Fraud Rate by Merchant Category (%)",
                      labels={"pct":"Fraud %","merchant_category":"Category"},
                      color="pct", color_continuous_scale="Reds")
        fig5.update_traces(textposition="outside")
        st.plotly_chart(fig5, use_container_width=True)

    # Income vs Loan Amount scatter
    st.markdown("---")
    fig6 = px.scatter(filtered_loans, x="monthly_income", y="loan_amount_requested",
                      color="fraud_flag", opacity=0.5,
                      title="Income vs Loan Amount (Fraud Highlighted)",
                      labels={"monthly_income":"Monthly Income",
                              "loan_amount_requested":"Loan Amount",
                              "fraud_flag":"Fraud Flag"})
    st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML Predictions & Model Performance
# ════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Predictions & Model Performance":
    st.title("🤖 ML Predictions & Model Performance")

    if not has_ml:
        st.warning(
            "ML outputs not found. Run `python pipeline_runner.py` to train the models first.\n\n"
            "Or click **Re-run Pipeline** in the sidebar."
        )
        st.stop()

    # ── Model Performance Cards ──────────────────────────────────────────
    st.markdown("### 📊 Model Performance Summary")

    if model_metrics:
        cols = st.columns(len(model_metrics))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i, (col, m) in enumerate(zip(cols, model_metrics)):
            with col:
                st.markdown(f"""
                <div style="background:{colors[i]};border-radius:10px;padding:16px;color:white;">
                  <h4 style="margin:0 0 8px 0;">{m['model']}</h4>
                  <p style="margin:2px 0;">Accuracy : <b>{m['accuracy']*100:.1f}%</b></p>
                  <p style="margin:2px 0;">Precision: <b>{m['precision']*100:.1f}%</b></p>
                  <p style="margin:2px 0;">Recall   : <b>{m['recall']*100:.1f}%</b></p>
                  <p style="margin:2px 0;">F1 Score : <b>{m['f1']*100:.1f}%</b></p>
                  <p style="margin:2px 0;">ROC-AUC  : <b>{m['roc_auc']:.4f}</b></p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Metric Comparison Bar Chart ──────────────────────────────────────
    st.markdown("### 📈 Model Comparison")
    if model_metrics:
        metric_names = ["accuracy","precision","recall","f1","roc_auc"]
        comparison_rows = []
        for m in model_metrics:
            for mn in metric_names:
                comparison_rows.append({"Model": m["model"], "Metric": mn.upper(),
                                        "Score": round(m[mn], 4)})
        comp_df = pd.DataFrame(comparison_rows)
        fig_comp = px.bar(comp_df, x="Metric", y="Score", color="Model", barmode="group",
                          title="Model Performance Comparison",
                          labels={"Score":"Score (0–1)"},
                          text=comp_df["Score"].apply(lambda x: f"{x:.3f}"))
        fig_comp.update_traces(textposition="outside")
        fig_comp.update_yaxes(range=[0, 1.1])
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # ── Loan Fraud Predictions ───────────────────────────────────────────
    st.markdown("### 🏦 Loan Fraud Predictions")
    if loan_pred is not None:
        lp = loan_pred.copy()
        lp_filtered = lp[lp["application_id"].isin(filtered_loans["application_id"])] \
                      if len(filtered_loans) < len(loan_df) else lp

        l1, l2, l3 = st.columns(3)
        l1.metric("Total Loans Scored",  f"{len(lp_filtered):,}")
        l2.metric("Predicted Fraudulent", f"{lp_filtered['predicted_fraud'].sum():,}")
        predicted_rate = lp_filtered['predicted_fraud'].mean() * 100 if len(lp_filtered) else 0
        l3.metric("Predicted Fraud Rate", f"{predicted_rate:.2f}%")

        # Probability distribution
        fig_dist = px.histogram(lp_filtered, x="fraud_probability",
                                color="predicted_fraud",
                                nbins=50,
                                title="Loan Fraud Probability Distribution",
                                labels={"fraud_probability":"Fraud Probability (%)",
                                        "predicted_fraud":"Predicted Fraud"})
        st.plotly_chart(fig_dist, use_container_width=True)

        # High risk loans table
        st.markdown("#### ⚠️ Top 20 Highest-Risk Loan Applications")
        high_risk = lp_filtered.sort_values("fraud_probability", ascending=False).head(20)
        high_risk_display = high_risk.merge(
            loan_df[["application_id","loan_type","loan_amount_requested",
                     "employment_status","cibil_score","loan_status"]],
            on="application_id", how="left"
        )
        high_risk_display["fraud_probability"] = high_risk_display["fraud_probability"].apply(
            lambda x: f"{x:.1f}%")
        st.dataframe(
            high_risk_display[["application_id","loan_type","loan_amount_requested",
                                "employment_status","cibil_score","loan_status",
                                "fraud_probability","predicted_fraud"]],
            use_container_width=True
        )

    st.markdown("---")

    # ── Transaction Fraud Predictions ────────────────────────────────────
    if txn_pred is not None and has_txns:
        st.markdown("### 💳 Transaction Fraud Predictions")

        tp = txn_pred.copy()
        tp_filtered = tp[tp["customer_id"].isin(filtered_txns["customer_id"])] \
                      if has_txns else tp

        t1, t2, t3 = st.columns(3)
        t1.metric("Total Transactions Scored",  f"{len(tp_filtered):,}")
        t2.metric("Predicted Fraudulent",        f"{tp_filtered['predicted_fraud'].sum():,}")
        txn_pred_rate = tp_filtered['predicted_fraud'].mean() * 100 if len(tp_filtered) else 0
        t3.metric("Predicted Fraud Rate",        f"{txn_pred_rate:.2f}%")

        # Fraud probability distribution
        fig_tdist = px.histogram(tp_filtered, x="fraud_probability",
                                 color="predicted_fraud", nbins=50,
                                 title="Transaction Fraud Probability Distribution",
                                 labels={"fraud_probability":"Fraud Probability (%)",
                                         "predicted_fraud":"Predicted Fraud"})
        st.plotly_chart(fig_tdist, use_container_width=True)

        # Actual vs Predicted fraud breakdown
        comp_cols = st.columns(2)
        with comp_cols[0]:
            actual_dist = tp_filtered["fraud_flag"].value_counts().reset_index()
            actual_dist.columns = ["fraud_flag","count"]
            actual_dist["label"] = actual_dist["fraud_flag"].map({0:"Legitimate",1:"Fraudulent"})
            fig_act = px.pie(actual_dist, values="count", names="label",
                             title="Actual Fraud Distribution", hole=0.4)
            st.plotly_chart(fig_act, use_container_width=True)
        with comp_cols[1]:
            pred_dist = tp_filtered["predicted_fraud"].value_counts().reset_index()
            pred_dist.columns = ["predicted_fraud","count"]
            pred_dist["label"] = pred_dist["predicted_fraud"].map({0:"Legitimate",1:"Fraudulent"})
            fig_pred = px.pie(pred_dist, values="count", names="label",
                              title="Predicted Fraud Distribution", hole=0.4)
            st.plotly_chart(fig_pred, use_container_width=True)

        # Top 20 high risk transactions
        st.markdown("#### ⚠️ Top 20 Highest-Risk Transactions")
        high_risk_txn = tp_filtered.sort_values("fraud_probability", ascending=False).head(20)
        high_risk_txn_display = high_risk_txn.merge(
            txn_df[["transaction_id","transaction_type","transaction_amount",
                    "merchant_category","device_used","transaction_status"]],
            on="transaction_id", how="left"
        )
        high_risk_txn_display["fraud_probability"] = high_risk_txn_display["fraud_probability"].apply(
            lambda x: f"{x:.1f}%")
        st.dataframe(
            high_risk_txn_display[["transaction_id","transaction_type","transaction_amount",
                                   "merchant_category","device_used","transaction_status",
                                   "fraud_probability","predicted_fraud"]],
            use_container_width=True
        )

    st.markdown("---")

    # ── Customer Risk Scores ─────────────────────────────────────────────
    if risk_df is not None:
        st.markdown("### 👤 Customer Risk Scores (Combined Model)")
        rd = risk_df.copy()

        r1, r2, r3 = st.columns(3)
        high_risk_customers = rd[rd["risk_score"] >= 50]
        r1.metric("Total Customers Scored", f"{len(rd):,}")
        r2.metric("High Risk (≥50%)",        f"{len(high_risk_customers):,}")
        r3.metric("Avg Risk Score",           f"{rd['risk_score'].mean():.1f}%")

        # Risk score distribution
        fig_risk = px.histogram(rd, x="risk_score", nbins=50,
                                title="Customer Risk Score Distribution",
                                labels={"risk_score":"Risk Score (%)"},
                                color_discrete_sequence=["#e74c3c"])
        fig_risk.add_vline(x=50, line_dash="dash", line_color="black",
                           annotation_text="High Risk Threshold")
        st.plotly_chart(fig_risk, use_container_width=True)

        # Risk bands
        rd["risk_band"] = pd.cut(
            rd["risk_score"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["Very Low (0-20)","Low (20-40)","Medium (40-60)",
                    "High (60-80)","Critical (80-100)"]
        )
        risk_band_df = rd["risk_band"].value_counts().reset_index()
        risk_band_df.columns = ["risk_band","count"]
        fig_bands = px.bar(risk_band_df, x="risk_band", y="count",
                           title="Customers by Risk Band",
                           color="risk_band",
                           color_discrete_sequence=["#2ecc71","#f39c12","#e67e22","#e74c3c","#8e44ad"],
                           labels={"risk_band":"Risk Band","count":"Number of Customers"})
        st.plotly_chart(fig_bands, use_container_width=True)

        # Top 20 highest risk customers
        st.markdown("#### 🚨 Top 20 Highest Risk Customers")
        top_risk = rd.sort_values("risk_score", ascending=False).head(20)
        top_risk_display = top_risk[["customer_id_original","risk_score","fraud_flag"]].copy()
        top_risk_display.columns = ["Customer ID","Risk Score (%)","Actual Fraud"]
        top_risk_display["Risk Score (%)"] = top_risk_display["Risk Score (%)"].apply(
            lambda x: f"{x:.1f}%")
        st.dataframe(top_risk_display, use_container_width=True)
