"""
train_models.py  --  Step 1 of the QuickLoan dashboard.

Trains every model in the bake-off (3 singles + 4 ensemble families), scores
them ALL on the SAME held-out test set, and writes the files the dashboard reads:

    leaderboard.json     -> metrics for every model (accuracy, recall, money saved...)
    all_models.joblib    -> every trained model (dashboard lets you pick one)
    loan_model.joblib    -> the champion (highest money saved) in the format app.py expects

Run:  python train_models.py
"""
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.metrics import accuracy_score, recall_score

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:  # brief allows HistGradientBoosting as a substitute
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False

INTEREST = 0.20          # brief: interest ~ 20% of the loan
SEED = 1                 # MUST match app.py so the dashboard's money card uses the same test set

# ------------------------------------------------------------------ 1. load / encode / split
df = pd.read_csv("loans.csv")
X = pd.get_dummies(df.drop(columns="default"), columns=["home_ownership"])
y = df["default"]

# Same split call as app.py (test_size, random_state, stratify) -> no leakage, same test rows
X_tr, X_te, y_tr, y_te, loan_tr, loan_te = train_test_split(
    X, y, df["loan_amount"], test_size=0.25, random_state=SEED, stratify=y)


def money_saved(y_true, y_pred, loan, interest=INTEREST):
    """Exact payoff logic from the brief."""
    y_true, y_pred, loan = map(np.array, (y_true, y_pred, loan))
    caught = (y_pred == 1) & (y_true == 1)   # defaulter rejected  -> loan saved
    wrong = (y_pred == 1) & (y_true == 0)    # good borrower rejected -> interest lost
    return loan[caught].sum() - interest * loan[wrong].sum()


# ------------------------------------------------------------------ 2. weak single models
def scaled(model):
    return make_pipeline(StandardScaler(), model)

dt = DecisionTreeClassifier(max_depth=3, random_state=SEED)
lr = scaled(LogisticRegression(max_iter=1000))
knn = scaled(KNeighborsClassifier(n_neighbors=15))

# ------------------------------------------------------------------ 3. strong learners reused in ensembles
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED)
if HAVE_XGB:
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
                        colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1)
else:
    xgb = HistGradientBoostingClassifier(random_state=SEED)

MODELS = {   # name: (family, estimator)
    "Decision Tree":     ("single",   dt),
    "Logistic Reg":      ("single",   lr),
    "KNN":               ("single",   knn),
    # Voting: different KINDS of models, soft vote (average probabilities)
    "Voting":            ("Voting",   VotingClassifier(
                              [("dt", DecisionTreeClassifier(max_depth=3, random_state=SEED)),
                               ("lr", scaled(LogisticRegression(max_iter=1000))),
                               ("knn", scaled(KNeighborsClassifier(n_neighbors=15)))], voting="soft")),
    # Bagging: many copies of the same model on bootstrap samples
    "Bagging":           ("Bagging",  BaggingClassifier(DecisionTreeClassifier(max_depth=6),
                                                         n_estimators=100, random_state=SEED, n_jobs=-1)),
    "Random Forest":     ("Bagging",  rf),
    # Boosting: models in sequence, each fixing previous mistakes
    "AdaBoost":          ("Boosting", AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED)),
    "Gradient Boosting": ("Boosting", gb),
    "XGBoost":           ("Boosting", xgb),
    # Stacking: a simple meta-model (logistic regression) learns how to trust strong base models
    "Stacking":          ("Stacking", StackingClassifier(
                              [("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
                               ("gb", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=SEED)),
                               ("lr", scaled(LogisticRegression(max_iter=1000)))],
                              final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1)),
}

# ------------------------------------------------------------------ 4. train + score on the SAME test set
rows, fitted = [], {}
yt = y_te.values
for name, (family, est) in MODELS.items():
    est.fit(X_tr, y_tr)
    pred = est.predict(X_te)
    caught = (pred == 1) & (yt == 1)
    wrong = (pred == 1) & (yt == 0)
    rows.append(dict(
        name=name, family=family,
        acc=round(float(accuracy_score(yt, pred)), 4),
        recall=round(float(recall_score(yt, pred)), 4),
        tp_n=int(caught.sum()), fp_n=int(wrong.sum()), fn_n=int(((pred == 0) & (yt == 1)).sum()),
        s_caught=float(loan_te.values[caught].sum()),     # AED of loans saved
        s_wrong=float(loan_te.values[wrong].sum()),       # AED of loans wrongly rejected (x interest = lost)
        money=float(money_saved(yt, pred, loan_te)),
    ))
    fitted[name] = est
    print(f"{name:18s} acc={rows[-1]['acc']:.3f}  recall={rows[-1]['recall']:.3f}  money=AED {rows[-1]['money']:>12,.0f}")

champion = max(rows, key=lambda r: r["money"])["name"]
ref = dict(total_def_loans=float(loan_te.values[yt == 1].sum()),
           total_good_loans=float(loan_te.values[yt == 0].sum()),
           n_def=int((yt == 1).sum()), n_good=int((yt == 0).sum()), n_test=int(len(yt)))

with open("leaderboard.json", "w") as f:
    json.dump({"models": rows, "ref": ref, "champion": champion, "interest": INTEREST}, f, indent=2)

COLUMNS = list(X_tr.columns)
joblib.dump({"models": fitted, "columns": COLUMNS}, "all_models.joblib", compress=3)
joblib.dump({"model": fitted[champion], "columns": COLUMNS, "name": champion}, "loan_model.joblib", compress=3)
print(f"\nChampion: {champion}\nSaved: leaderboard.json, all_models.joblib, loan_model.joblib")
