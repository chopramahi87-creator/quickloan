# app.py -- QuickLoan Risk Dashboard (upgraded from the provided starter)
# Run:  python train_models.py   (once)   then   streamlit run app.py
# Needs in the same folder: loans.csv, leaderboard.json, all_models.joblib, loan_model.joblib
import json
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="QuickLoan Risk Dashboard", page_icon="💰", layout="wide")

# ----------------------------------------------------------------------------- loaders
@st.cache_resource
def load_models():
    bundle = joblib.load("all_models.joblib")
    return bundle["models"], bundle["columns"]

@st.cache_data
def load_board():
    with open("leaderboard.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("loans.csv")
    return df

import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)   # always read/write files next to app.py

if not os.path.exists("all_models.joblib"):
    if not os.path.exists("loans.csv") or not os.path.exists("train_models.py"):
        st.error("Put app.py, train_models.py and loans.csv in the SAME folder, then rerun.")
        st.stop()
    with st.spinner("First run: training all models (about 1-2 minutes)..."):
        r = subprocess.run([sys.executable, "train_models.py"], capture_output=True, text=True)
    if r.returncode != 0:
        st.error("Training failed. Run `python train_models.py` in a terminal to see the error.")
        st.code(r.stderr[-1500:])
        st.stop()

try:
    MODELS, COLUMNS = load_models()
    BOARD = load_board()
    DF = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run `python train_models.py` first, in this folder.")
    st.stop()

CHAMPION = BOARD["champion"]

def encode(df):
    return pd.get_dummies(df, columns=["home_ownership"]).reindex(columns=COLUMNS, fill_value=0)

def money_saved(y_true, y_pred, loan, interest=0.20):
    y_true, y_pred, loan = np.array(y_true), np.array(y_pred), np.array(loan)
    caught = (y_pred == 1) & (y_true == 1)     # defaulter rejected -> loan saved
    wrong = (y_pred == 1) & (y_true == 0)      # good borrower rejected -> interest lost
    return loan[caught].sum() - interest * loan[wrong].sum()

@st.cache_data
def test_split():
    # Identical to train_models.py -> the held-out rows were never seen in training
    X = encode(DF.drop(columns="default")); y = DF["default"]
    _, Xte, _, yte, _, lo_te = train_test_split(
        X, y, DF["loan_amount"], test_size=0.25, random_state=1, stratify=y)
    return Xte, yte, lo_te

def fmt(n):
    return f"{'−' if n < 0 else ''}AED {abs(round(n)):,}"

# ----------------------------------------------------------------------------- sidebar (global controls)
st.sidebar.title("⚙️ Controls")
interest = st.sidebar.slider("Interest rate (cost of rejecting a good customer)", 0.05, 0.50, 0.20, 0.01,
                             format="%.2f")
model_name = st.sidebar.selectbox("Model used in Decision & Portfolio tabs",
                                  list(MODELS), index=list(MODELS).index(CHAMPION))
threshold = st.sidebar.slider("Reject if default probability ≥", 0.05, 0.95, 0.50, 0.01,
                              help="0.50 is the default. Lowering it rejects more people -> catches more "
                                   "defaulters (recall up) but also rejects more good borrowers.")
break_even = interest / (1 + interest)
st.sidebar.info(f"💡 Break-even threshold at {interest:.0%} interest ≈ **{break_even:.2f}**. "
                f"Rejecting only pays off when P(default) is above that.")
model = MODELS[model_name]

st.title("💰 QuickLoan — Risk Dashboard")
st.caption("One expert can be wrong. A committee wins. Judge every model by **money saved**, not accuracy.")

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboard", "🧾 Loan Decision", "💵 Portfolio Money", "📊 Data Explorer"])

# ============================================================================= TAB 1: leaderboard
with tab1:
    lb = pd.DataFrame(BOARD["models"])
    lb["money"] = lb["s_caught"] - interest * lb["s_wrong"]          # recomputed for chosen interest
    lb = lb.sort_values("money", ascending=False).reset_index(drop=True)
    winner = lb.iloc[0]

    k1, k2, k3 = st.columns(3)
    k1.metric("Accuracy spread (best − worst)", f"{(lb.acc.max() - lb.acc.min()) * 100:.1f} pts",
              help="Tiny: accuracy cannot tell the models apart.")
    k2.metric("Money-saved spread (best − worst)", fmt(lb.money.max() - lb.money.min()))
    k3.metric("Champion", winner["name"], winner["family"] if winner["family"] != "single" else "single model")

    c1, c2 = st.columns(2)
    color = alt.Color("family:N", title="Type", scale=alt.Scale(
        domain=["single", "Voting", "Bagging", "Boosting", "Stacking"],
        range=["#94a3b8", "#8b5cf6", "#2563eb", "#f59e0b", "#14b8a6"]))
    with c1:
        st.subheader("Accuracy — all look the same")
        st.altair_chart(alt.Chart(lb).mark_bar().encode(
            y=alt.Y("name:N", sort="-x", title=None),
            x=alt.X("acc:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Accuracy"),
            color=color, tooltip=["name", "family", alt.Tooltip("acc:Q", format=".1%")]
        ).properties(height=330), use_container_width=True)
    with c2:
        st.subheader("Money saved vs approve-all")
        st.altair_chart(alt.Chart(lb).mark_bar().encode(
            y=alt.Y("name:N", sort="-x", title=None),
            x=alt.X("money:Q", title="AED saved"),
            color=color, tooltip=["name", "family", alt.Tooltip("money:Q", format=",.0f")]
        ).properties(height=330), use_container_width=True)

    st.subheader("Full leaderboard")
    show = lb[["name", "family", "acc", "recall", "tp_n", "fp_n", "fn_n", "money"]].rename(columns={
        "name": "Model", "family": "Type", "acc": "Accuracy", "recall": "Recall (defaulters caught)",
        "tp_n": "Defaulters caught", "fp_n": "Good rejected", "fn_n": "Defaulters missed", "money": "Money saved (AED)"})
    st.dataframe(show.style.format({"Accuracy": "{:.1%}", "Recall (defaulters caught)": "{:.1%}",
                                    "Money saved (AED)": "{:,.0f}"}), use_container_width=True, hide_index=True)

    ref = BOARD["ref"]
    reject_all = ref["total_def_loans"] - interest * ref["total_good_loans"]
    st.caption(f"Reference — approve everyone: AED 0 · reject everyone: {fmt(reject_all)} · "
               f"test set: {ref['n_test']} loans ({ref['n_def']} defaulters). Leaderboard uses the default 0.5 cut-off.")
    st.info(f"**The lesson:** accuracy differs by only {(lb.acc.max() - lb.acc.min()) * 100:.1f} points, but money "
            f"saved differs by {fmt(lb.money.max() - lb.money.min())}. Recall on defaulters drives the money.")

# ============================================================================= TAB 2: decision card
with tab2:
    st.subheader(f"Loan decision — model: {model_name}")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 21, 64, 35)
        annual_income = st.number_input("Annual income (AED)", 24000, 400000, 90000, step=1000)
        loan_amount = st.number_input("Loan amount (AED)", 5000, 150000, 40000, step=1000)
        loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
        employment_years = st.slider("Employment years", 0.0, 35.0, 4.0, 0.5)
    with c2:
        credit_history_months = st.slider("Credit history (months)", 0, 300, 60)
        num_existing_loans = st.slider("Existing loans", 0, 8, 1)
        past_defaults = st.slider("Past defaults", 0, 6, 0)
        home_ownership = st.selectbox("Home ownership", ["rent", "own", "mortgage"])
    debt_to_income = round(loan_amount / (annual_income + 1), 2)
    st.caption(f"Debt-to-income (auto): {debt_to_income}")

    applicant = dict(age=age, annual_income=annual_income, loan_amount=loan_amount,
                     loan_term_months=loan_term_months, employment_years=employment_years,
                     credit_history_months=credit_history_months, num_existing_loans=num_existing_loans,
                     past_defaults=past_defaults, debt_to_income=debt_to_income, home_ownership=home_ownership)
    X1 = encode(pd.DataFrame([applicant]))
    prob = float(model.predict_proba(X1)[0][1])
    reject = prob >= threshold

    d1, d2, d3 = st.columns(3)
    with d1:
        (st.error if reject else st.success)(
            f"{'❌ REJECT' if reject else '✅ APPROVE'} — default risk {prob * 100:.0f}%")
    d2.metric("Default probability", f"{prob * 100:.1f}%", f"cut-off {threshold:.0%}", delta_color="off")
    ev_approve = (1 - prob) * interest * loan_amount - prob * loan_amount
    d3.metric("Expected profit if approved", fmt(ev_approve),
              help="(1−p)·interest·loan − p·loan. Negative means approving is expected to lose money.")
    st.progress(min(prob, 1.0), text="Risk meter")

# ============================================================================= TAB 3: portfolio money
with tab3:
    st.subheader(f"Portfolio view — {model_name} on the held-out test set")
    Xte, yte, lo_te = test_split()
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    yt = yte.values
    caught = int(((pred == 1) & (yt == 1)).sum()); total_def = int((yt == 1).sum())
    wrong = int(((pred == 1) & (yt == 0)).sum()); missed = total_def - caught
    saved_amt = lo_te.values[(pred == 1) & (yt == 1)].sum()
    lost_int = interest * lo_te.values[(pred == 1) & (yt == 0)].sum()
    net = saved_amt - lost_int
    missed_amt = lo_te.values[(pred == 0) & (yt == 1)].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💵 Money saved vs approve-all", fmt(net))
    m2.metric("🎯 Defaulters caught", f"{caught}/{total_def}", f"recall {caught / total_def:.0%}", delta_color="off")
    m3.metric("😕 Good borrowers rejected", wrong)
    m4.metric("Accuracy", f"{(pred == yt).mean():.1%}")

    st.markdown(f"**This model would save QuickLoan {fmt(net)} vs. approving everyone.**")
    wf = pd.DataFrame({"Item": ["✅ Loans saved (defaulters caught)", "👎 Interest lost (good rejected)",
                                "= Net money saved", "💸 Still lost (defaulters missed)"],
                       "AED": [saved_amt, -lost_int, net, -missed_amt]})
    st.altair_chart(alt.Chart(wf).mark_bar().encode(
        y=alt.Y("Item:N", sort=None, title=None), x=alt.X("AED:Q"),
        color=alt.Color("Item:N", legend=None, scale=alt.Scale(
            range=["#16a34a", "#d97706", "#2563eb", "#dc2626"])),
        tooltip=["Item", alt.Tooltip("AED:Q", format=",.0f")]).properties(height=200), use_container_width=True)

    # money vs threshold curve -> shows where the best cut-off is
    ts = np.arange(0.05, 0.96, 0.01)
    curve = pd.DataFrame({"threshold": ts,
                          "money": [money_saved(yt, (proba >= t).astype(int), lo_te, interest) for t in ts]})
    best = curve.loc[curve.money.idxmax()]
    st.subheader("Money saved vs cut-off")
    base = alt.Chart(curve).mark_line(color="#2563eb").encode(
        x=alt.X("threshold:Q", title="Reject if P(default) ≥"), y=alt.Y("money:Q", title="Money saved (AED)"))
    mark = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="#f59e0b", strokeDash=[4, 4]).encode(x="t:Q")
    st.altair_chart(base + mark, use_container_width=True)
    st.caption(f"Best cut-off on this test set ≈ {best.threshold:.2f} → {fmt(best.money)} "
               f"(orange line = your current cut-off). Tuned on the test set, so treat it as indicative.")

# ============================================================================= TAB 4: data explorer
with tab4:
    st.subheader("What drives default?")
    e1, e2, e3 = st.columns(3)
    e1.metric("Loans in dataset", f"{len(DF):,}")
    e2.metric("Overall default rate", f"{DF.default.mean():.1%}")
    e3.metric("Avg loan", fmt(DF.loan_amount.mean()))

    g1, g2 = st.columns(2)
    with g1:
        by = DF.groupby("past_defaults").default.mean().reset_index()
        st.altair_chart(alt.Chart(by).mark_bar(color="#dc2626").encode(
            x=alt.X("past_defaults:O", title="Past defaults"),
            y=alt.Y("default:Q", axis=alt.Axis(format="%"), title="Default rate")
        ).properties(title="Default rate by past defaults", height=260), use_container_width=True)
    with g2:
        by = DF.groupby("home_ownership").default.mean().reset_index()
        st.altair_chart(alt.Chart(by).mark_bar(color="#2563eb").encode(
            x=alt.X("home_ownership:N", title="Home ownership"),
            y=alt.Y("default:Q", axis=alt.Axis(format="%"), title="Default rate")
        ).properties(title="Default rate by home ownership", height=260), use_container_width=True)

    DF2 = DF.assign(dti_band=pd.cut(DF.debt_to_income, [0, .3, .6, .9, 1.2, 3.01]).astype(str))
    by = DF2.groupby("dti_band").default.mean().reset_index()
    st.altair_chart(alt.Chart(by).mark_bar(color="#f59e0b").encode(
        x=alt.X("dti_band:N", sort=None, title="Debt-to-income band"),
        y=alt.Y("default:Q", axis=alt.Axis(format="%"), title="Default rate")
    ).properties(title="Default rate by debt-to-income", height=260), use_container_width=True)

    # feature importance (tree-based models only)
    est = model
    imp = getattr(est, "feature_importances_", None)
    if imp is None and hasattr(est, "estimators_") and hasattr(est.estimators_[0], "feature_importances_"):
        try:
            imp = np.mean([e.feature_importances_ for e in est.estimators_], axis=0)
        except Exception:
            imp = None
    if imp is not None and len(imp) == len(COLUMNS):
        fi = pd.DataFrame({"feature": COLUMNS, "importance": imp}).sort_values("importance", ascending=False)
        st.altair_chart(alt.Chart(fi).mark_bar(color="#14b8a6").encode(
            y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("importance:Q")
        ).properties(title=f"Feature importance — {model_name}", height=300), use_container_width=True)
    else:
        st.caption(f"Feature importance isn't available for {model_name} — pick a tree-based model in the sidebar.")
