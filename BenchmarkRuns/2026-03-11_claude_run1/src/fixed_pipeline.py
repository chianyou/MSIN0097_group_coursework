"""
Task E: FIXED pipeline.
Corrects all 4 bugs from broken_pipeline.py:
  FIX-1: fraud_label excluded from features
  FIX-2: transaction_id and user_id excluded from features
  FIX-3: Scaler fitted on TRAINING data only (via sklearn Pipeline)
  FIX-4: Stratified split with fixed random seed
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskA_data_audit" / "outputs" / "cleaned.csv"
OUT_DIR = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskE_bug_leakage_debug"
OUTPUTS = OUT_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42  # FIX-4: fixed seed
TARGET = "fraud_label"
ID_COLS = ["transaction_id", "user_id"]  # FIX-2: will be excluded

df = pd.read_csv(CLEANED)

# FIX-1 + FIX-2: Drop target AND ID columns from feature matrix
X = df.drop(columns=[TARGET] + ID_COLS)
y = df[TARGET]

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# FIX-4: Stratified split
X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y   # stratified!
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_tr_full, y_tr_full, test_size=0.15 / 0.85,
    random_state=RANDOM_SEED, stratify=y_tr_full
)

# FIX-3: Preprocessing fitted ONLY on X_train (inside Pipeline)
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
    )),
])

pipeline.fit(X_train, y_train)   # scaler sees only training rows

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

metrics_fixed = {
    "pipeline": "fixed",
    "fixes_applied": [
        "FIX-1: fraud_label excluded from feature matrix",
        "FIX-2: transaction_id and user_id excluded from feature matrix",
        "FIX-3: StandardScaler fitted on training data only (sklearn Pipeline)",
        "FIX-4: Stratified train_test_split with random_state=42",
    ],
    "split": {
        "train": len(y_train),
        "valid": len(y_valid),
        "test": len(y_test),
    },
    "PR_AUC":    round(float(average_precision_score(y_test, y_prob)), 6),
    "ROC_AUC":   round(float(roc_auc_score(y_test, y_prob)), 6),
    "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
    "Recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
    "F1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

print("=== FIXED Pipeline Metrics ===")
for k, v in metrics_fixed.items():
    if k not in ("fixes_applied", "confusion_matrix", "split"):
        print(f"  {k}: {v}")

with open(OUTPUTS / "metrics_fixed.json", "w") as f:
    json.dump(metrics_fixed, f, indent=2)

print("Saved: metrics_fixed.json")
print("FIXED pipeline complete.")
