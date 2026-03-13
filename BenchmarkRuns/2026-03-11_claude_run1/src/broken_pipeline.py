"""
Task E: BROKEN pipeline (deliberately flawed).
Contains 4 deliberate bugs:
  BUG-1: Target leakage  — fraud_label included as a feature
  BUG-2: IDs used as predictors — transaction_id, user_id kept in X
  BUG-3: Preprocessing fit on FULL data — scaler sees test labels before split
  BUG-4: Non-stratified split — stratify=None (ignores class imbalance)

Run this script to reproduce the inflated/invalid metrics.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskA_data_audit" / "outputs" / "cleaned.csv"
OUT_DIR = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskE_bug_leakage_debug"
OUTPUTS = OUT_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEANED)

# BUG-2: Keep ID columns in features
# BUG-1: Keep fraud_label IN X (target leakage)
# Encode all object columns naively
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=[])          # BUG-1: fraud_label NOT dropped — still in X!
y = df["fraud_label"]

# BUG-3: Fit scaler on FULL dataset (before split) — leaks test statistics into training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # fit on all data including test rows
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# BUG-4: Non-stratified split — random split ignores class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=None  # BUG-4: no stratify, no fixed seed
)

# Fit logistic regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate on test set
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

metrics_broken = {
    "pipeline": "broken",
    "bugs": [
        "BUG-1: fraud_label included in features (target leakage)",
        "BUG-2: transaction_id, user_id used as predictors",
        "BUG-3: StandardScaler fitted on full dataset before split",
        "BUG-4: train_test_split with no stratify and no random_state",
    ],
    "PR_AUC":    round(float(average_precision_score(y_test, y_prob)), 6),
    "ROC_AUC":   round(float(roc_auc_score(y_test, y_prob)), 6),
    "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
    "Recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
    "F1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

print("=== BROKEN Pipeline Metrics ===")
for k, v in metrics_broken.items():
    if k not in ("bugs", "confusion_matrix"):
        print(f"  {k}: {v}")

with open(OUTPUTS / "metrics_broken.json", "w") as f:
    json.dump(metrics_broken, f, indent=2)

print("Saved: metrics_broken.json")
print("BROKEN pipeline complete.")
